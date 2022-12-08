#ifndef __SBI_STEP_H__
#define __SBI_STEP_H__

#include <sbi/sbi_trap.h>

int sbi_step_enabled();

void sbi_step_enable();
void sbi_step_disable();
void sbi_step_breakpoint(struct sbi_trap_regs* regs);
void sbi_step_interrupt(struct sbi_trap_regs* regs, uintptr_t handler);

#endif
