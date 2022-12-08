#include <sbi/sbi_step.h>
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

static uintptr_t pte_pa(pte_t* pte) {
	return ((uintptr_t)pte->ppn0 << 12) | ((uintptr_t)pte->ppn1 << 21) | ((uintptr_t)pte->ppn2 << 30);
}

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

static int pte_leaf(pte_t* pte) {
	return pte->valid && (pte->read || pte->write || pte->exec);
}

typedef struct {
	pte_t ptes[512];
} pagetable_t;

/* static ulong prev_mideleg; */
/* static ulong prev_medeleg; */
/* static int prev_tvm; */

static uintptr_t vpn(int level, uintptr_t va) {
	return (va >> (12+9*level)) & ((1UL << 9) - 1);
}

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
	INSN_EBREAK = 0x00100073,
	INSN_SRET = 0x10200073,
};

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

#define OP(x) bits_get(x, 6, 0)
#define RD(x) bits_get(x, 11, 7)
#define RS1(x) bits_get(x, 19, 15)
#define RS2(x) bits_get(x, 24, 20)
#define SHAMT(x) bits_get(x, 24, 20)
#define FUNCT3(x) bits_get(x, 14, 12)
#define FUNCT7(x) bits_get(x, 31, 25)

static uint32_t extract_imm(uint32_t insn, imm_type_t type) {
    switch (type) {
        case IMM_I:
            return sext(bits_remap(insn, 31, 20, 11, 0), 12);
        case IMM_S:
            return sext(
                bits_remap(insn, 11, 7, 4, 0) | bits_remap(insn, 31, 25, 11, 5),
                12);
        case IMM_B:
            return sext(bit_remap(insn, 7, 11) | bits_remap(insn, 11, 8, 4, 1) |
                            bits_remap(insn, 30, 25, 10, 5) |
                            bit_remap(insn, 31, 12),
                        13);
        case IMM_J:
            return sext(
                bit_remap(insn, 31, 20) | bits_remap(insn, 30, 21, 10, 1) |
                    bit_remap(insn, 20, 11) | bits_remap(insn, 19, 12, 19, 12),
                21);
        case IMM_U:
            return insn & ~bits_mask(12);
    }
    assert(0);
	return 0;
}

void sbi_step_breakpoint(struct sbi_trap_regs *regs) {
	uint32_t* epc = (uint32_t*) epcpa(regs->mepc);
	sbi_printf("sbi_step_breakpoint, epc: %p\n", epc);

	if (epc != brkpt) {
		sbi_printf("ERROR: epc != brkpt\n");
		while (1) {}
	}

	// replace current breakpoint with orig bytes
	*brkpt = insn;

	pagetable_t* pt = get_pt();

	// decode instruction to decide where to jump next
	uintptr_t nextva;
	unsigned long* regsidx = (unsigned long*) regs;
	switch (OP(insn)) {
		case OP_JAL:
			nextva = regs->mepc + extract_imm(insn, IMM_J);
			break;
		case OP_JALR:
			nextva = regsidx[RS1(insn)] + extract_imm(insn, IMM_I);
			break;
		case OP_BRANCH:
			uintptr_t res = 0;
			switch (FUNCT3(insn)) {
				case 0b000:
				case 0b001:
					res = regsidx[RS1(insn)] ^ regsidx[RS2(insn)];
					break;
				case 0b100:
				case 0b101:
					res = (intptr_t)regsidx[RS1(insn)] < (intptr_t)regsidx[RS2(insn)];
				case 0b110:
				case 0b111:
					res = regsidx[RS1(insn)] < regsidx[RS2(insn)];
			}
			int cond = false;
			switch (FUNCT3(insn)) {
				case 0b000:
				case 0b101:
				case 0b111:
					cond = res == 0;
				case 0b001:
				case 0b100:
				case 0b110:
					cond = res != 0;
			}
			if (cond) {
				nextva = regs->mepc + extract_imm(insn, IMM_B);
			} else {
				nextva = regs->mepc + 4;
			}
			break;
		default:
			if (insn == INSN_SRET) {
				nextva = csr_read(CSR_SEPC);
			} else {
				nextva = regs->mepc + 4;
			}
	}
	uint32_t* next = (uint32_t*) va2pa(pt, nextva, NULL);
	sbi_printf("nextva: %lx, nextpa: %p\n", nextva, next);

	// place breakpoint there
	brkpt = next;
	insn = *brkpt;
	*brkpt = INSN_EBREAK;

	RISCV_FENCE_I;
}

static void sbi_ecall_step_enable(const struct sbi_trap_regs *regs) {
    _sbi_step_enabled = 1;
	uintptr_t next = epcpa(regs->mepc) + 4;
	// place breakpoint at EPC+4
	brkpt = (uint32_t*) next;
	insn = *brkpt;
	*brkpt = INSN_EBREAK;
	RISCV_FENCE_I;
	// TODO: disable trap delegation in mideleg and medeleg
}

static void sbi_ecall_step_disable(const struct sbi_trap_regs *regs) {
    _sbi_step_enabled = 0;

	// TODO: reset mideleg and medeleg

	// remove breakpoint
	*brkpt = insn;
	RISCV_FENCE_I;
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
	}
	return ret;
}

struct sbi_ecall_extension ecall_step = {
    .extid_start = SBI_EXT_STEP,
    .extid_end = SBI_EXT_STEP,
    .handle = sbi_ecall_step_handler,
};
