#include <sbi/sbi_step.h>
#include <sbi/sbi_hart.h>
#include <sbi/sbi_ecall.h>
#include <sbi/sbi_ecall_interface.h>
#include <sbi/sbi_error.h>
#include <sbi/sbi_trap.h>
#include <sbi/sbi_version.h>
#include <sbi/riscv_asm.h>
#include <sbi/riscv_barrier.h>
#include <sbi/sbi_console.h>
#include <sbi/bits.h>

static void assert(int cond) {
	if (!cond) {
		sbi_printf("assertion failed\n");
		while (1) {}
	}
}

static int _sbi_step_enabled;

int sbi_step_enabled() {
    return _sbi_step_enabled;
}

// Since OpenSBI runs in M-mode, which has virtual memory disabled, we have to
// manually convert VAs to PAs by walking the pagetable.
typedef struct {
	uintptr_t valid:1;
	uintptr_t read:1;
	uintptr_t write:1;
	uintptr_t exec:1;
	uintptr_t user:1;
	uintptr_t global:1;
	uintptr_t accessed:1;
	uintptr_t dirty:1;
	uintptr_t rsw:2;
	uintptr_t ppn0:9;
	uintptr_t ppn1:9;
	uintptr_t ppn2:26;
	uintptr_t _reserved:10;
} pte_t;
_Static_assert(sizeof(pte_t) == 8, "invalid size for pte_t");

// Returns the physical address stored in a pte.
static uintptr_t pte_pa(pte_t* pte) {
	return ((uintptr_t)pte->ppn0 << 12) | ((uintptr_t)pte->ppn1 << 21) | ((uintptr_t)pte->ppn2 << 30);
}

// Given a leaf pte and a va, converts the va to a pa.
static uintptr_t pte_va2pa(pte_t* pte, int level, uintptr_t va) {
	switch (level) {
		case 2:
			return ((uintptr_t)pte->ppn2 << 30) | (va & ((1UL << 30) - 1));
		case 1:
			return ((uintptr_t)pte->ppn1 << 30) | ((uintptr_t)pte->ppn1) | (va & ((1UL << 21) - 1));
		case 0:
			return pte_pa(pte) | (va & 0xfff);
	}
	return 0;
}

// Returns true if this is a leaf pte.
static int pte_leaf(pte_t* pte) {
	return pte->valid && (pte->read || pte->write || pte->exec);
}

typedef struct {
	pte_t ptes[512];
} pagetable_t;

// Returns the virtual page number of a va at a given pagetable level.
static uintptr_t vpn(int level, uintptr_t va) {
	return (va >> (12+9*level)) & ((1UL << 9) - 1);
}

// Converts a virtual address to physical address by walking the pagetable.
static uintptr_t va2pa(pagetable_t* pt, uintptr_t va, int* failed) {
	if (!pt) {
		return va;
	}

	int endlevel = 0;
	for (int level = 2; level > endlevel; level--) {
		pte_t* pte = &pt->ptes[vpn(level, va)];
		if (pte_leaf(pte)) {
			return pte_va2pa(pte, level, va);
		} else if (pte->valid) {
			pt = (pagetable_t*) pte_pa(pte);
		} else {
			if (failed)
				*failed = true;
			return 0;
		}
	}
	return pte_va2pa(&pt->ptes[vpn(endlevel, va)], endlevel, va);
}

// Gets the currently active pagetable.
static pagetable_t* get_pt() {
	uintptr_t satp = csr_read(CSR_SATP);
	assert((satp >> 60) == 8);
	return (pagetable_t*) (satp << 12);
}

static uintptr_t epcpa(uintptr_t mepc) {
	pagetable_t* pt = get_pt();
	uintptr_t epc = va2pa(pt, mepc, NULL);
	return epc;
}

static uint32_t* brkpt;
static uint32_t insn;

enum {
	INSN_ECALL = 0x00000073,
	INSN_EBREAK = 0x00100073,
	INSN_SRET = 0x10200073,
};

static void place_breakpoint(uint32_t* loc) {
	insn = *loc;
	brkpt = loc;
	*brkpt = INSN_EBREAK;
	RISCV_FENCE_I;
}

typedef enum {
    OP_RARITH = 0b0110011,
    OP_IARITH = 0b0010011,
    OP_BRANCH = 0b1100011,
    OP_LUI    = 0b0110111,
    OP_AUIPC  = 0b0010111,
    OP_JAL    = 0b1101111,
    OP_JALR   = 0b1100111,
    OP_LOAD   = 0b0000011,
    OP_STORE  = 0b0100011,
    OP_FENCE  = 0b0001111,
    OP_SYS    = 0b1110011,
} op_t;

typedef enum {
    IMM_I,
    IMM_S,
    IMM_B,
    IMM_J,
    IMM_U,
} imm_type_t;

typedef enum {
    EXT_BYTE  = 0b000,
    EXT_HALF  = 0b001,
    EXT_WORD  = 0b010,
	EXT_DWORD = 0b011,
    EXT_BYTEU = 0b100,
    EXT_HALFU = 0b101,
	EXT_WORDU = 0b110,
} imm_ext_t;

#define OP(x) bits_get(x, 6, 0)
#define RD(x) bits_get(x, 11, 7)
#define RS1(x) bits_get(x, 19, 15)
#define RS2(x) bits_get(x, 24, 20)
#define SHAMT(x) bits_get(x, 24, 20)
#define FUNCT3(x) bits_get(x, 14, 12)
#define FUNCT7(x) bits_get(x, 31, 25)

static uint64_t extract_imm(uint32_t insn, imm_type_t type) {
    switch (type) {
        case IMM_I:
            return sext64(bits_remap(insn, 31, 20, 11, 0), 12);
        case IMM_S:
            return sext64(
                bits_remap(insn, 11, 7, 4, 0) | bits_remap(insn, 31, 25, 11, 5),
                12);
        case IMM_B:
            return sext64(bit_remap(insn, 7, 11) | bits_remap(insn, 11, 8, 4, 1) |
                            bits_remap(insn, 30, 25, 10, 5) |
                            bit_remap(insn, 31, 12),
                        13);
        case IMM_J:
            return sext64(
                bit_remap(insn, 31, 20) | bits_remap(insn, 30, 21, 10, 1) |
                    bit_remap(insn, 20, 11) | bits_remap(insn, 19, 12, 19, 12),
                21);
        case IMM_U:
            return insn & ~bits_mask(12);
    }
    assert(0);
	return 0;
}

typedef struct {
	char* start;
	char* end;
	bool active;
} dev_region_t;

int n_dev_regions;
#define MAX_DEV_REGION 10
dev_region_t dev_regions[MAX_DEV_REGION];
int dev_active;

dev_region_t text_region;

typedef struct {
	void* start;
	size_t sz;
	bool active;
} heap_t;

heap_t heap;

/* FENCE CHECKER */
/* static void on_fence_i() { */
/*  */
/* } */

static void on_fence_dev() {
	dev_regions[dev_active].active = false;
}

/* static void on_fence_vma() { */
/*  */
/* } */

static void on_load(char* addr, size_t sz) {
	for (int i = 0; i < n_dev_regions; i++) {
		if (i == dev_active && dev_regions[dev_active].active)
			continue;
		if (addr >= dev_regions[i].start && addr <= dev_regions[i].end) {
			if (dev_regions[dev_active].active) {
				sbi_printf("ERROR: load from inactive device region\n");
				sbi_hart_hang();
			}
			dev_active = i;
			dev_regions[dev_active].active = true;
		}
	}
}

static void on_store(char* addr, size_t sz) {
	if (text_region.active && addr >= text_region.start && addr <= text_region.end) {
		sbi_printf("ERROR: store to text segment\n");
		sbi_hart_hang();
	}

	for (int i = 0; i < n_dev_regions; i++) {
		if (i == dev_active && dev_regions[dev_active].active)
			continue;
		if (addr >= dev_regions[i].start && addr <= dev_regions[i].end) {
			if (dev_regions[dev_active].active) {
				sbi_printf("ERROR: store to inactive device region\n");
				sbi_hart_hang();
			}
			dev_active = i;
			dev_regions[dev_active].active = true;
		}
	}
}

/* static void on_execute(void* addr) { */
/*  */
/* } */

static void sbi_ecall_step_devfence_region(void* start, void* end) {
	if (n_dev_regions >= MAX_DEV_REGION) {
		return;
	}
	dev_regions[n_dev_regions++] = (dev_region_t){
		.start = (char*) start,
		.end = (char*) end,
	};
}

void sbi_step_interrupt(struct sbi_trap_regs *regs, uintptr_t handlerva) {
	uint32_t* handler = (uint32_t*) epcpa(handlerva);
	sbi_printf("sbi_step_interrupt, handler: %p\n", handler);
	// put a breakpoint at the start of the interrupt handler
	place_breakpoint(handler);
}

// Called when we reach a breakpoint with single stepping enabled. We should
// move the breakpoint one instruction further and continue. If the current
// instruction is a jump, branch, or sret then we need to calculate what
// address the instruction will jump to next (by partially evaluting the
// instruction), and put the breakpoint there.
void sbi_step_breakpoint(struct sbi_trap_regs *regs) {
	uint32_t* epc = (uint32_t*) epcpa(regs->mepc);
	/* sbi_printf("sbi_step_breakpoint, epc: %p\n", epc); */

	if (epc != brkpt) {
		sbi_printf("ERROR: epc != brkpt\n");
		while (1) {}
	}

	unsigned long* regsidx = (unsigned long*) regs;

	size_t imm;
	char* addr;
	switch (OP(insn)) {
		case OP_FENCE:
			on_fence_dev();
			break;
		case OP_LOAD:
			imm = extract_imm(insn, IMM_I);
			addr = (char*) (regsidx[RS1(insn)] + imm);

			switch (FUNCT3(insn)) {
				case EXT_BYTEU:
				case EXT_BYTE:
					on_load(addr, 1);
					break;
				case EXT_HALFU:
				case EXT_HALF:
					on_load(addr, 2);
					break;
				case EXT_WORDU:
				case EXT_WORD:
					on_load(addr, 4);
					break;
				case EXT_DWORD:
					on_load(addr, 8);
					break;
				default:
					assert(0);
			}
			break;
		case OP_STORE:
			imm = extract_imm(insn, IMM_S);
			addr = (char*) (regsidx[RS1(insn)] + imm);

			switch (FUNCT3(insn)) {
				case EXT_BYTE:
					on_store(addr, 1);
					break;
				case EXT_HALF:
					on_store(addr, 2);
					break;
				case EXT_WORD:
					on_store(addr, 4);
					break;
				case EXT_DWORD:
					on_store(addr, 8);
					break;
				default:
					assert(0);
			}
			break;
	}

	// replace current breakpoint with orig bytes
	*brkpt = insn;
	RISCV_FENCE_I;

	pagetable_t* pt = get_pt();

	// decode instruction to decide where to jump next
	uintptr_t nextva;
	switch (OP(insn)) {
		case OP_JAL:
			nextva = regs->mepc + extract_imm(insn, IMM_J);
			break;
		case OP_JALR:
			nextva = regsidx[RS1(insn)] + extract_imm(insn, IMM_I);
			break;
		case OP_BRANCH:;
			uintptr_t res = 0;
			switch (FUNCT3(insn)) {
				case 0b000:
				case 0b001:
					res = regsidx[RS1(insn)] ^ regsidx[RS2(insn)];
					break;
				case 0b100:
				case 0b101:
					res = (intptr_t)regsidx[RS1(insn)] < (intptr_t)regsidx[RS2(insn)];
					break;
				case 0b110:
				case 0b111:
					res = regsidx[RS1(insn)] < regsidx[RS2(insn)];
					break;
			}
			int cond = false;
			switch (FUNCT3(insn)) {
				case 0b000:
				case 0b101:
				case 0b111:
					cond = res == 0;
					break;
				case 0b001:
				case 0b100:
				case 0b110:
					cond = res != 0;
					break;
			}
			if (cond) {
				nextva = regs->mepc + extract_imm(insn, IMM_B);
			} else {
				nextva = regs->mepc + 4;
			}
			break;
		default:
			switch (insn) {
				case INSN_SRET:
					nextva = csr_read(CSR_SEPC);
					break;
				case INSN_ECALL:
					nextva = csr_read(CSR_STVEC);
					break;
				default:
					nextva = regs->mepc + 4;
			}
	}
	uint32_t* next = (uint32_t*) va2pa(pt, nextva, NULL);
	/* sbi_printf("nextva: %lx, nextpa: %p\n", nextva, next); */

	// place breakpoint there
	place_breakpoint(next);
}

/* static ulong prev_mideleg; */
/* static ulong prev_medeleg; */

static void sbi_ecall_step_enable(const struct sbi_trap_regs *regs) {
    _sbi_step_enabled = 1;
	uintptr_t next = epcpa(regs->mepc) + 4;
	// place breakpoint at EPC+4
	place_breakpoint((uint32_t*) next);

	/* prev_mideleg = csr_read(CSR_MIDELEG); */
	/* prev_medeleg = csr_read(CSR_MEDELEG); */
	/* csr_write(CSR_MIDELEG, 0); */
	/* csr_write(CSR_MEDELEG, 0); */
}

static void sbi_ecall_step_disable(const struct sbi_trap_regs *regs) {
	// remove breakpoint
	if (_sbi_step_enabled) {
		*brkpt = insn;
		/* csr_write(CSR_MIDELEG, prev_mideleg); */
		/* csr_write(CSR_MEDELEG, prev_medeleg); */
	}

    _sbi_step_enabled = 0;

	RISCV_FENCE_I;
}

static void sbi_ecall_step_set_heap(void* heap_start, size_t sz) {
	heap = (heap_t){
		.active = true,
		.start = heap_start,
		.sz = sz,
	};
}

static int sbi_ecall_step_handler(unsigned long extid, unsigned long funcid, const struct sbi_trap_regs *regs, unsigned long *out_val, struct sbi_trap_info *out_trap) {
	int ret = 0;
	switch (funcid) {
		case SBI_EXT_STEP_ENABLED:
			ret = sbi_step_enabled();
			break;
		case SBI_EXT_STEP_ENABLE:
			sbi_ecall_step_enable(regs);
			break;
		case SBI_EXT_STEP_DISABLE:
			sbi_ecall_step_disable(regs);
			break;
		case SBI_EXT_STEP_TEXT_REGION:
			text_region.active = true;
			text_region.start = (char*) regs->a0;
			text_region.end = (char*) regs->a1;
			break;
		case SBI_EXT_STEP_DEVFENCE_REGION:
			sbi_ecall_step_devfence_region((void*) regs->a0, (void*) regs->a1);
			break;
		case SBI_EXT_STEP_SET_HEAP:
			sbi_ecall_step_set_heap((void*) regs->a0, (size_t) regs->a1);
			break;
	}
	return ret;
}

struct sbi_ecall_extension ecall_step = {
    .extid_start = SBI_EXT_STEP,
    .extid_end = SBI_EXT_STEP,
    .handle = sbi_ecall_step_handler,
};
