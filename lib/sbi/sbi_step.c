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
#include <sbi/sbi_step_ht.h>
#include <sbi/sbi_string.h>

void* kr_malloc(unsigned nbytes);
void kr_free(void* ptr);

static void assert(int cond) {
	if (!cond) {
		sbi_printf("assertion failed\n");
		while (1) {}
	}
}

static int _sbi_step_enabled;
static unsigned _sbi_flags;

#define FLAG_SET(val, flag) (((val & flag) != 0))
#define OPT_SET(flag) (((_sbi_flags & flag) != 0))

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

typedef struct ptestat {
	char* addr;
	bool modified;
	struct ptestat* next;
} ptestat_t;

static ptestat_t* ptestats;

static void free_ptestats() {
	ptestat_t* p = ptestats;
	ptestats = NULL;
	while (p) {
		ptestat_t* next = p->next;
		p->next = NULL;
		kr_free(p);
		p = next;
	}
}

static void load_pt_rec(pagetable_t* pt, pagetable_t* prev, bool modified) {
	for (int i = 0; i < 512; i++) {
		pte_t* pte = &pt->ptes[i];
		if (!prev || ((uint64_t*)pt->ptes)[i] != ((uint64_t*)prev->ptes)[i]) {
			if (!pte->valid) {
				continue;
			}
			ptestat_t* stat = (ptestat_t*) kr_malloc(sizeof(ptestat_t));
			if (!stat) {
				sbi_printf("ERROR: vmafence checker ran out of memory\n");
				sbi_hart_hang();
			}
			stat->addr = (char*) pte;
			stat->modified = modified;
			if (pte_leaf(pte) && (!pte->accessed || !pte->dirty)) {
				sbi_printf("ERROR: accessed/dirty bits were not set in leaf PTE\n");
				sbi_hart_hang();
			}
			stat->next = ptestats;
			ptestats = stat;
			if (pte->valid && !pte_leaf(pte)) {
				load_pt_rec((pagetable_t*) pte_pa(&pt->ptes[i]), NULL, modified);
			}
		} else if (pte->valid && !pte_leaf(pte)) {
			load_pt_rec((pagetable_t*) pte_pa(&pt->ptes[i]), (pagetable_t*) pte_pa(&prev->ptes[i]), modified);
		}
	}
}

static void load_pt(pagetable_t* pt, pagetable_t* prev, bool modified) {
	free_ptestats();
	load_pt_rec(pt, prev, modified);
}

static bool is_modified(char* addr) {
	ptestat_t* p = ptestats;
	while (p) {
		if (p->addr == addr) {
			return p->modified;
		}
		p = p->next;
	}
	return false;
}

// Returns the virtual page number of a va at a given pagetable level.
static uintptr_t vpn(int level, uintptr_t va) {
	return (va >> (12+9*level)) & ((1UL << 9) - 1);
}

// Converts a virtual address to physical address by walking the pagetable.
static uintptr_t va2pa(pagetable_t* pt, uintptr_t va, int* failed, bool check) {
	if (!pt) {
		return va;
	}

	int endlevel = 0;
	for (int level = 2; level > endlevel; level--) {
		pte_t* pte = &pt->ptes[vpn(level, va)];
		if (check && is_modified((char*) pte)) {
			sbi_printf("ERROR: va %lx, accessing a dirty pagetable entry without sfence.vma\n", va);
			sbi_hart_hang();
		}
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
	/* assert((satp >> 60) == 8); */
	return (pagetable_t*) (satp << 12);
}

static uintptr_t epcpa(uintptr_t mepc) {
	pagetable_t* pt = get_pt();
	uintptr_t epc = va2pa(pt, mepc, NULL, false);
	return epc;
}

enum {
	INSN_ECALL = 0x00000073,
	INSN_EBREAK = 0x00100073,
	INSN_SRET = 0x10200073,
};

static void place_breakpoint(uint32_t* loc) {
	csr_write(CSR_TSELECT, 0);
	unsigned mcontrol = 0b011100;
	csr_write(CSR_TDATA1, mcontrol);
	csr_write(CSR_TDATA2, loc);
}

static void place_breakpoint_mismatch(uint32_t* loc) {
	csr_write(CSR_TSELECT, 0);
	csr_write(CSR_TDATA1, bits_set(0b011100, 10, 7, 2));
	csr_write(CSR_TDATA2, (uintptr_t) loc + 2);

	csr_write(CSR_TSELECT, 1);
	csr_write(CSR_TDATA1, bits_set(0b011100, 10, 7, 3));
	csr_write(CSR_TDATA2, (uintptr_t) loc);
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
#define CSR(x) bits_get(x, 31, 20)

enum {
	SATP_NUM = 0x180,
};

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

typedef struct region {
	char* start;
	size_t sz;
	struct region* next;
} region_t;

static region_t* dev_regions;
static region_t* dev_active;

static region_t* rdonly_regions;
static region_t* noexec_regions;

static void free_regions(region_t** regions) {
	region_t* r = *regions;
	*regions = NULL;
	while (r) {
		region_t* next = r->next;
		r->next = NULL;
		kr_free(r);
		r = next;
	}
}

typedef struct {
	char* start;
	char* brk;
	size_t sz;
	bool active;
} heap_t;

static char heap_data[1024 * 16];
heap_t heap = (heap_t){
	.start = heap_data,
	.brk = heap_data,
	.sz = sizeof(heap_data),
	.active = true,
};

ht_t fence_ht[2];
ht_t mem_ht;

static void on_fence_i() {
	if (OPT_SET(SS_IFENCE)) {
		// clear all unflushed addresses
		uintptr_t id = csr_read(CSR_MHARTID);
		sbi_memset(&fence_ht[id].entries[0], 0, sizeof(ht_entry_t) * fence_ht[id].cap);
		fence_ht[id].len = 0;
	}
}

static void on_fence_dev() {
	if (OPT_SET(SS_REGION)) {
		dev_active = NULL;
	}
}

static void on_fence_vma(unsigned rs1, unsigned rs2, struct sbi_trap_regs* regs) {
	if (!OPT_SET(SS_VMAFENCE)) {
		return;
	}

	ptestat_t* p = ptestats;
	while (p) {
		p->modified = false;
		p = p->next;
	}
}

static void on_satp_write(ulong value) {
	if (!OPT_SET(SS_VMAFENCE)) {
		return;
	}

	load_pt((pagetable_t*) (value << 12), get_pt(), true);
}

static void on_load(char* addr, size_t sz) {
	if (OPT_SET(SS_VMAFENCE)) {
		// walk the address: make sure that each pt page touched is not modified
		va2pa(get_pt(), (uintptr_t) addr, NULL, true);
	}
	if (OPT_SET(SS_REGION)) {
		region_t* r = dev_regions;
		while (r) {
			if (r != dev_active) {
				if (addr >= r->start && addr < r->start + r->sz) {
					if (dev_active) {
						sbi_printf("ERROR: load from inactive device region\n");
						sbi_hart_hang();
					}
					dev_active = r;
					break;
				}
			}
			r = r->next;
		}
	}
	if (OPT_SET(SS_MEM)) {
		bool found;
		for (int i = 0; i < mem_ht.len; i++) {
			ht_entry_t* ent = &mem_ht.entries[i];
			if (ent->filled && addr >= (char*) ent->key && addr < (char*) ent->key + ent->val) {
				found = true;
				break;
			}
		}
		if (!found) {
			sbi_printf("ERROR: load from unallocated region\n");
			sbi_hart_hang();
		}
	}
}

static void on_store(char* addr, size_t sz, uint64_t stval) {
	if (OPT_SET(SS_VMAFENCE)) {
		ptestat_t* p = ptestats;
		// if modifying a ptpage: need to mark the modified page as modified
		while (p) {
			if (p->addr >= addr && p->addr < addr + sz) {
				p->modified = true;
				if (!bit_get(stval, 7) || !bit_get(stval, 6)) {
					sbi_printf("ERROR: accessed/dirty bits were not set in PTE\n");
					sbi_hart_hang();
				}
			}
			p = p->next;
		}
		// if the page was leaf: modified-leaf
		// if the page was global: modified-global
		// walk the address
		va2pa(get_pt(), (uintptr_t) addr, NULL, true);
	}

	if (OPT_SET(SS_IFENCE)) {
		if (ht_put(&fence_ht[0], (uint64_t) epcpa((uintptr_t) addr), true) == -1) {
			sbi_printf("ERROR: ifence checker ran out of memory\n");
			sbi_hart_hang();
		}
		if (ht_put(&fence_ht[1], (uint64_t) epcpa((uintptr_t) addr), true) == -1) {
			sbi_printf("ERROR: ifence checker ran out of memory\n");
			sbi_hart_hang();
		}
	}

	if (OPT_SET(SS_REGION)) {
		region_t* r = rdonly_regions;
		while (r) {
			if (addr >= r->start && addr < r->start + r->sz) {
				sbi_printf("ERROR: store to read-only region\n");
				sbi_hart_hang();
			}
			r = r->next;
		}

		r = dev_regions;
		while (r) {
			if (r != dev_active) {
				if (addr >= r->start && addr < r->start + r->sz) {
					if (dev_active) {
						sbi_printf("ERROR: store to inactive device region\n");
						sbi_hart_hang();
					}
					dev_active = r;
					break;
				}
			}
			r = r->next;
		}
	}

	if (OPT_SET(SS_MEM)) {
		bool found = false;
		for (int i = 0; i < mem_ht.cap; i++) {
			ht_entry_t* ent = &mem_ht.entries[i];
			if (ent->filled && addr >= (char*) ent->key && addr < (char*) ent->key + ent->val) {
				found = true;
				break;
			}
		}
		if (!found) {
			sbi_printf("ERROR: store to unallocated region %p\n", addr);
			sbi_hart_hang();
		}
	}
}

static uint64_t equiv_hash = 0xdeadbeef;

static void on_execute(char* va, char* addr, struct sbi_trap_regs* regs) {
	if (OPT_SET(SS_VMAFENCE)) {
		// walk the address
		va2pa(get_pt(), (uintptr_t) va, NULL, true);
	}

	if (OPT_SET(SS_REGION)) {
		region_t* r = noexec_regions;
		while (r) {
			if (addr >= r->start && addr < r->start + r->sz) {
				sbi_printf("ERROR: attempt to execute noexec region\n");
				sbi_hart_hang();
			}
			r = r->next;
		}
	}

	if (OPT_SET(SS_IFENCE)) {
		uintptr_t id = csr_read(CSR_MHARTID);
		if (ht_get(&fence_ht[id], (uint64_t) addr, NULL)) {
			sbi_printf("ERROR: attempt to execute unfenced instruction\n");
			sbi_hart_hang();
		}
	}

	if (OPT_SET(SS_EQUIV)) {
		uint64_t hash = ht_hash(equiv_hash ^ (uint64_t) addr);
		uint64_t* regsidx = (uint64_t*) regs;
		for (int i = 0; i < 32; i++) {
			hash = ht_hash(hash ^ regsidx[i]);
		}
		equiv_hash = hash;
	}
}

// Called when we reach a breakpoint with single stepping enabled. We should
// move the breakpoint one instruction further and continue. If the current
// instruction is a jump, branch, or sret then we need to calculate what
// address the instruction will jump to next (by partially evaluting the
// instruction), and put the breakpoint there.
void sbi_step_breakpoint(struct sbi_trap_regs* regs) {
	/* sbi_printf("sbi_step_breakpoint, epc: %lx, mtval: %lx\n", regs->mepc, csr_read(CSR_MTVAL)); */
	uint32_t* epc = (uint32_t*) epcpa(regs->mepc);

	uint32_t insn = *epc;

	unsigned long* regsidx = (unsigned long*) regs;

	on_execute((char*) regs->mepc, (char*) epc, regs);

	size_t imm;
	char* addr;
	switch (OP(insn)) {
		case OP_FENCE:
			switch(FUNCT3(insn)) {
				case 0b000:
					on_fence_dev();
					break;
				case 0b001:
					on_fence_i();
					break;
			}
			break;
		case OP_SYS:
			if (FUNCT7(insn) == 0b0001001 && FUNCT3(insn) == 0b000) {
				on_fence_vma(RS1(insn), RS2(insn), regs);
			}
			if (FUNCT3(insn) == 0b001 && CSR(insn) == SATP_NUM) {
				on_satp_write(regsidx[RS1(insn)]);
			}
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
			uint64_t stval = regsidx[RS2(insn)];

			switch (FUNCT3(insn)) {
				case EXT_BYTE:
					on_store(addr, 1, stval);
					break;
				case EXT_HALF:
					on_store(addr, 2, stval);
					break;
				case EXT_WORD:
					on_store(addr, 4, stval);
					break;
				case EXT_DWORD:
					on_store(addr, 8, stval);
					break;
				default:
					assert(0);
			}
			break;
	}

	place_breakpoint_mismatch((uint32_t*) regs->mepc);
}

void sbi_ecall_step_enable_at(uintptr_t addr, unsigned flags) {
	_sbi_step_enabled = 1;
	_sbi_flags = flags;
	// place breakpoint at EPC+4
	place_breakpoint((uint32_t*) addr);

	if (OPT_SET(SS_IFENCE)) {
		uintptr_t id = csr_read(CSR_MHARTID);
		if (ht_alloc(&fence_ht[id], 128) == -1) {
			sbi_printf("ERROR: could not allocate fence hashtable\n");
			sbi_hart_hang();
		}
	}

	if (OPT_SET(SS_VMAFENCE)) {
		load_pt(get_pt(), NULL, false);
	}
}

static void sbi_ecall_step_disable(const struct sbi_trap_regs *regs) {
	// remove breakpoint
	if (_sbi_step_enabled) {
		csr_write(CSR_TSELECT, 0);
		csr_write(CSR_TDATA1, 0);
		csr_write(CSR_TSELECT, 1);
		csr_write(CSR_TDATA1, 0);
	}

	if (OPT_SET(SS_IFENCE)) {
		/* ht_free(&fence_ht[id]); */
	}

	if (OPT_SET(SS_REGION)) {
		free_regions(&dev_regions);
		dev_active = NULL;
		free_regions(&rdonly_regions);
		free_regions(&noexec_regions);
	}

    _sbi_step_enabled = 0;

	/* RISCV_FENCE_I; */
}

static void sbi_ecall_step_mark_region(void* start, size_t sz, unsigned flags) {
	if (FLAG_SET(flags, SS_REGION_DEVICE)) {
		region_t* r = (region_t*) kr_malloc(sizeof(region_t));
		r->next = dev_regions;
		r->start = start;
		r->sz = sz;
		dev_regions = r;
	}
	if (FLAG_SET(flags, SS_REGION_RDONLY)) {
		region_t* r = (region_t*) kr_malloc(sizeof(region_t));
		r->next = rdonly_regions;
		r->start = start;
		r->sz = sz;
		rdonly_regions = r;
	}
	if (FLAG_SET(flags, SS_REGION_NOEXEC)) {
		region_t* r = (region_t*) kr_malloc(sizeof(region_t));
		r->next = noexec_regions;
		r->start = start;
		r->sz = sz;
		noexec_regions = r;
	}
	if (!mem_ht.entries) {
		if (ht_alloc(&mem_ht, 128) == -1) {
			sbi_printf("ERROR: could not allocate memory hashtable\n");
			sbi_hart_hang();
		}
	}
	if (ht_put(&mem_ht, (uint64_t) start, (uint64_t) sz) == -1) {
		sbi_printf("ERROR: could not allocate memory for memory checker\n");
		sbi_hart_hang();
	}
}

static void sbi_ecall_step_mem_alloc(void* ptr, size_t sz) {
	if (!mem_ht.entries) {
		if (ht_alloc(&mem_ht, 128) == -1) {
			sbi_printf("ERROR: could not allocate memory hashtable\n");
			sbi_hart_hang();
		}
	}
	if (ht_put(&mem_ht, (uint64_t) ptr, (uint64_t) sz) == -1) {
		sbi_printf("ERROR: could not allocate memory for memory checker\n");
		sbi_hart_hang();
	}
}
static void sbi_ecall_step_mem_free(void* ptr) {
	ht_remove(&mem_ht, (uint64_t) ptr);
}

static int sbi_ecall_step_equiv_hash() {
	return equiv_hash;
}

static int sbi_ecall_step_handler(unsigned long extid, unsigned long funcid, const struct sbi_trap_regs *regs, unsigned long *out_val, struct sbi_trap_info *out_trap) {
	int ret = 0;
	switch (funcid) {
		case SBI_EXT_STEP_ENABLED:
			ret = sbi_step_enabled();
			break;
		case SBI_EXT_STEP_ENABLE:
			sbi_ecall_step_enable_at(regs->mepc + 4, regs->a0);
			break;
		case SBI_EXT_STEP_ENABLE_AT:
			sbi_ecall_step_enable_at(regs->a0, regs->a1);
			break;
		case SBI_EXT_STEP_DISABLE:
			sbi_ecall_step_disable(regs);
			break;
		case SBI_EXT_STEP_MEM_ALLOC:
			sbi_ecall_step_mem_alloc((void*) regs->a0, (size_t) regs->a1);
			break;
		case SBI_EXT_STEP_MEM_FREE:
			sbi_ecall_step_mem_free((void*) regs->a0);
			break;
		case SBI_EXT_STEP_MARK_REGION:
			sbi_ecall_step_mark_region((void*) regs->a0, (size_t) regs->a1, regs->a2);
			break;
		case SBI_EXT_STEP_EQUIV_HASH:
			ret = sbi_ecall_step_equiv_hash();
			break;
	}
	return ret;
}

struct sbi_ecall_extension ecall_step = {
    .extid_start = SBI_EXT_STEP,
    .extid_end = SBI_EXT_STEP,
    .handle = sbi_ecall_step_handler,
};
