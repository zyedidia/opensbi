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

typedef struct {
	char* start;
	char* brk;
	size_t sz;
	bool active;
} heap_t;
extern heap_t heap;

void* kr_malloc(unsigned nbytes);
void kr_free(void* ptr);

#define roundup(x,n) (((x)+((n)-1))&(~((n)-1)))

union align {
	double d;
	void *p;
	void (*fp)(void);
};

typedef union header { /* block header */
	struct {
		union header *ptr; /* next block if on free list */
		unsigned size; /* size of this block */
	} s;
	union align x; /* force alignment of blocks */
} header_t;

static header_t base; /* empty list to get started */
static header_t *freep = NULL; /* start of free list */

void* sbrk(size_t increment) {
	if (heap.brk >= heap.start + heap.sz) {
		return NULL;
	}
	void* p = heap.brk;
	heap.brk += increment;
	return p;
}

#define NALLOC 1024 /* minimum #units to request */
/* morecore: ask system for more memory */
static header_t* morecore(unsigned nu) {
	if (!heap.active)
		return NULL;
	char *cp;
	header_t *up;
	if (nu < NALLOC)
		nu = NALLOC;
	cp = sbrk(nu * sizeof(header_t));
	if (cp == NULL) /* no space at all */
		return NULL;
	up = (header_t *) cp;
	up->s.size = nu;
	kr_free((void *)(up+1));
	return freep;
}

void* kr_malloc(unsigned nbytes) {
	header_t *p, *prevp;
	header_t *morecore(unsigned);
	unsigned nunits;
	nunits = (nbytes+sizeof(header_t)-1)/sizeof(header_t) + 1;
	if ((prevp = freep) == NULL) { /* no free list yet */
		base.s.ptr = freep = prevp = &base;
		base.s.size = 0;
	}
	for (p = prevp->s.ptr; ; prevp = p, p = p->s.ptr) {
		if (p->s.size >= nunits) { /* big enough */
			if (p->s.size == nunits) /* exactly */
				prevp->s.ptr = p->s.ptr;
			else { /* allocate tail end */
				p->s.size -= nunits;
				p += p->s.size;
				p->s.size = nunits;
			}
			freep = prevp;
			return (void *)(p+1);
		}
		if (p == freep) /* wrapped around free list */
			if ((p = morecore(nunits)) == NULL)
				return NULL; /* none left */
	}
}

/* free: put block ap in free list */
void kr_free(void* ap) {
	header_t *bp, *p;
	bp = (header_t *)ap - 1; /* point to block header */
	for (p = freep; !(bp > p && bp < p->s.ptr); p = p->s.ptr)
		if (p >= p->s.ptr && (bp > p || bp < p->s.ptr))
			break; /* freed block at start or end of arena */
	if (bp + bp->s.size == p->s.ptr) { /* join to upper nbr */
		bp->s.size += p->s.ptr->s.size;
		bp->s.ptr = p->s.ptr->s.ptr;
	} else
		bp->s.ptr = p->s.ptr;
	if (p + p->s.size == bp) { /* join to lower nbr */
		p->s.size += bp->s.size;
		p->s.ptr = bp->s.ptr;
	} else
		p->s.ptr = bp;
	freep = p;
}
