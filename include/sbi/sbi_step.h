#ifndef __SBI_STEP_H__
#define __SBI_STEP_H__

#include <sbi/sbi_trap.h>

int sbi_step_enabled();

enum {
    SS_IFENCE = (1 << 0),
    SS_VMAFENCE = (1 << 1),
    SS_REGION = (1 << 2),
    SS_MEM = (1 << 3),
    SS_EQUIV = (1 << 4),
    SS_NOSTEP_ECALL = (1 << 5),
};

enum {
    SS_REGION_DEVICE = (1 << 0),
    SS_REGION_RDONLY = (1 << 1),
    SS_REGION_NOEXEC = (1 << 2),
    SS_REGION_STACK = (1 << 3),
    SS_REGION_RDWR = (1 << 4),
};

void sbi_step_disable();
void sbi_step_breakpoint(struct sbi_trap_regs* regs);
void sbi_step_interrupt(struct sbi_trap_regs* regs, uintptr_t handler);

#endif
