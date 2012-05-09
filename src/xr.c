/*-
 * Copyright (c) 2012 Ilya Kaliman
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <stdlib.h>
#include <string.h>

#include "efp_private.h"
#include "disp.h"

static inline int
fock_idx(int i, int j)
{
	return i < j ?
		j * (j + 1) / 2 + i :
		i * (i + 1) / 2 + j;
}

static inline double
valence(double n)
{
	if (n > 2)
		n -= 2;
	if (n > 8)
		n -= 8;
	return n;
}

static inline double
get_disp_damp_overlap(double s_ij)
{
	double ln_s = log(fabs(s_ij));
	return 1.0 - s_ij * s_ij * (1.0 - 2.0 * ln_s + 2.0 * ln_s * ln_s);
}

static void
calc_disp_damp_overlap(struct efp *efp, int frag_i, int frag_j,
		       int i, int j, double s_ij)
{
	if (!efp->disp_damp_overlap)
		return;

	double damp = fabs(s_ij) > 1.0e-6 ? get_disp_damp_overlap(s_ij) : 0.0;

	efp->disp_damp_overlap[
		disp_damp_overlap_idx(efp, frag_i, frag_j, i, j)] = damp;
	efp->disp_damp_overlap[
		disp_damp_overlap_idx(efp, frag_j, frag_i, j, i)] = damp;
}

static inline double
get_charge_pen(double s_ij, double r_ij)
{
	double ln_s = log(fabs(s_ij));
	return -2.0 * s_ij * s_ij / r_ij / sqrt(-2.0 * ln_s);
}

static inline int
get_block_frag_count(struct efp *efp, int i)
{
	return efp->xr_block_frag_offset[i + 1] - efp->xr_block_frag_offset[i];
}

static double
compute_xr_frag(struct efp *efp, int frag_i, int frag_j, int offset,
		int stride, struct efp_st_data *st)
{
	struct frag *fr_i = efp->frags + frag_i;
	struct frag *fr_j = efp->frags + frag_j;

	double lmo_s[fr_i->n_lmo][fr_j->n_lmo];
	double lmo_t[fr_i->n_lmo][fr_j->n_lmo];
	double tmp[fr_i->n_lmo * fr_j->xr_wf_size];

	/* lmo_s = wf_i * s * wf_j(t) */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		fr_i->n_lmo, fr_j->xr_wf_size, fr_i->xr_wf_size, 1.0,
		fr_i->xr_wf, fr_i->xr_wf_size, st->s + offset, stride, 0.0,
		tmp, fr_j->xr_wf_size);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		fr_i->n_lmo, fr_j->n_lmo, fr_j->xr_wf_size, 1.0,
		tmp, fr_j->xr_wf_size, fr_j->xr_wf, fr_j->xr_wf_size, 0.0,
		&lmo_s[0][0], fr_j->n_lmo);

	/* lmo_t = wf_i * t * wf_j(t) */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		fr_i->n_lmo, fr_j->xr_wf_size, fr_i->xr_wf_size, 1.0,
		fr_i->xr_wf, fr_i->xr_wf_size, st->t + offset, stride, 0.0,
		tmp, fr_j->xr_wf_size);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		fr_i->n_lmo, fr_j->n_lmo, fr_j->xr_wf_size, 1.0,
		tmp, fr_j->xr_wf_size, fr_j->xr_wf, fr_j->xr_wf_size, 0.0,
		&lmo_t[0][0], fr_j->n_lmo);

	double energy = 0.0;

	for (int i = 0; i < fr_i->n_lmo; i++) {
		for (int j = 0; j < fr_j->n_lmo; j++) {
			double s_ij = lmo_s[i][j];
			double t_ij = lmo_t[i][j];
			double r_ij = vec_dist(fr_i->lmo_centroids + i,
					       fr_j->lmo_centroids + j);

			calc_disp_damp_overlap(efp, frag_i, frag_j, i, j, s_ij);
			efp->charge_pen_energy += get_charge_pen(s_ij, r_ij);

			/* xr - first part */
			if (fabs(s_ij) > 1.0e-6)
				energy += -2.0 * sqrt(-2.0 * log(fabs(s_ij)) /
						PI) * s_ij * s_ij / r_ij;

			/* xr - second part */
			for (int k = 0; k < fr_i->n_lmo; k++)
				energy -= s_ij * lmo_s[k][j] *
					fr_i->xr_fock_mat[fock_idx(i, k)];
			for (int l = 0; l < fr_j->n_lmo; l++)
				energy -= s_ij * lmo_s[i][l] *
					fr_j->xr_fock_mat[fock_idx(j, l)];
			energy += 2.0 * s_ij * t_ij;

			/* xr - third part */
			for (int jj = 0; jj < fr_j->n_atoms; jj++) {
				struct efp_atom *at = fr_j->atoms + jj;
				double r = vec_dist(fr_i->lmo_centroids + i,
							VEC(at->x));
				energy -= s_ij * s_ij * valence(at->znuc) / r;
			}
			for (int l = 0; l < fr_j->n_lmo; l++) {
				double r = vec_dist(fr_i->lmo_centroids + i,
						    fr_j->lmo_centroids + l);
				energy += 2.0 * s_ij * s_ij / r;
			}
			for (int ii = 0; ii < fr_i->n_atoms; ii++) {
				struct efp_atom *at = fr_i->atoms + ii;
				double r = vec_dist(fr_j->lmo_centroids + j,
							VEC(at->x));
				energy -= s_ij * s_ij * valence(at->znuc) / r;
			}
			for (int k = 0; k < fr_i->n_lmo; k++) {
				double r = vec_dist(fr_i->lmo_centroids + k,
						    fr_j->lmo_centroids + j);
				energy += 2.0 * s_ij * s_ij / r;
			}
			energy -= s_ij * s_ij / r_ij;
		}
	}
	energy *= 2.0;
	return energy;
}

static enum efp_result
compute_xr_block(struct efp *efp, int block_i, int block_j, double *energy)
{
	enum efp_result res;

	struct frag *frags_i = efp->frags + efp->xr_block_frag_offset[block_i];
	struct frag *frags_j = efp->frags + efp->xr_block_frag_offset[block_j];
	int n_block_i = get_block_frag_count(efp, block_i);
	int n_block_j = get_block_frag_count(efp, block_j);

	struct efp_st_block block;
	memset(&block, 0, sizeof(struct efp_st_block));

	for (int i = 0; i < n_block_i; i++) {
		block.n_atoms_i += frags_i[i].n_atoms;
		block.basis_size_i += frags_i[i].xr_wf_size;
	}

	for (int j = 0; j < n_block_j; j++) {
		block.n_atoms_j += frags_j[j].n_atoms;
		block.basis_size_j += frags_j[j].xr_wf_size;
	}

	block.atoms_i = malloc(block.n_atoms_i * sizeof(struct efp_atom));
	block.atoms_j = malloc(block.n_atoms_j * sizeof(struct efp_atom));

	for (int i = 0, idx = 0; i < n_block_i; i++)
		for (int a = 0; a < frags_i[i].n_atoms; a++)
			block.atoms_i[idx++] = frags_i[i].atoms[a];

	for (int j = 0, idx = 0; j < n_block_j; j++)
		for (int a = 0; a < frags_j[j].n_atoms; a++)
			block.atoms_j[idx++] = frags_j[j].atoms[a];

	size_t size = block.basis_size_i * block.basis_size_j * sizeof(double);

	struct efp_st_data st;
	memset(&st, 0, sizeof(struct efp_st_data));

	st.s = malloc((efp->grad ? 4 : 1) * size);
	st.t = malloc((efp->grad ? 4 : 1) * size);

	if (efp->grad) {
		st.sx = st.s + 1 * size;
		st.sy = st.s + 2 * size;
		st.sz = st.s + 3 * size;
		st.tx = st.t + 1 * size;
		st.ty = st.t + 2 * size;
		st.tz = st.t + 3 * size;
	}

	if ((res = efp->callbacks.get_st_integrals(&block, (efp->grad ? 1 : 0),
			&st, efp->callbacks.get_st_integrals_user_data)))
		goto fail;

	for (int i = 0, offset = 0; i < n_block_i; i++) {
		int j = 0;

		if (block_i == block_j)
			for (; j < i + 1; j++)
				offset += frags_j[j].xr_wf_size;

		for (; j < n_block_j; j++) {
			int frag_i = i + efp->xr_block_frag_offset[block_i];
			int frag_j = j + efp->xr_block_frag_offset[block_j];

			*energy += compute_xr_frag(efp, frag_i, frag_j, offset,
						   block.basis_size_j, &st);

			offset += frags_j[j].xr_wf_size;
		}
		offset += (frags_i[i].xr_wf_size - 1) * block.basis_size_j;
	}

fail:
	free(block.atoms_i), free(block.atoms_j);
	free(st.s), free(st.t);
	return res;
}

enum efp_result
efp_compute_xr(struct efp *efp)
{
	if (efp->grad)
		return EFP_RESULT_NOT_IMPLEMENTED;

	enum efp_result res;
	double energy = 0.0;
	efp->charge_pen_energy = 0.0;

	/* Because of potentially huge number of fragments we can't just
	 * compute all overlap and kinetic energy integrals in one step.
	 * Instead we process fragments in blocks so that number of basis
	 * functions in each block is not greater than some reasonable number.
	 * Also see setup_xr function in efp.c for block setup details.
	 */
	for (int i = 0; i < efp->n_xr_blocks; i++)
		for (int j = i; j < efp->n_xr_blocks; j++)
			if ((res = compute_xr_block(efp, i, j, &energy)))
				return res;

	efp->energy[efp_get_term_index(EFP_TERM_XR)] = energy;
	return EFP_RESULT_SUCCESS;
}

static void
rotate_func_d(const mat_t *rotmat, const double *in, double *out)
{
	/* GAMESS order */
	enum { xx = 0, yy, zz, xy, xz, yz };

	const double norm = sqrt(3.0) / 2.0;

	/* XXX hack for now - input in GAMESS order output in Q-Chem order */
	double full_in[9] = {
		       in[xx], norm * in[xy], norm * in[xz],
		norm * in[xy],        in[yy], norm * in[yz],
		norm * in[xz], norm * in[yz],        in[zz]
	};

	double full_out[9];
	rotate_t2(rotmat, full_in, full_out);

	/* Q-Chem order */
	out[0] = full_out[0];
	out[1] = full_out[1] / norm;
	out[2] = full_out[4];
	out[3] = full_out[2] / norm;
	out[4] = full_out[5] / norm;
	out[5] = full_out[8];
}

static void
rotate_func_f(const mat_t *rotmat, const double *in, double *out)
{
	/* GAMESS order */
	enum { xxx = 0, xxy = 3, xyy = 5, yyy = 1, xxz = 4,
	       xyz = 9, yyz = 6, xzz = 7, yzz = 8, zzz = 2 };

	const double norm1 = sqrt(5.0) / 3.0;
	const double norm2 = sqrt(3.0) / 2.0;

	/* XXX hack for now - input in GAMESS order output in Q-Chem order */
	double full_in[27] = {
in[xxx],                 in[xxy] * norm1,         in[xxz] * norm1,
in[xxy] * norm1,         in[xyy] * norm1,         in[xyz] * norm1 * norm2,
in[xxz] * norm1,         in[xyz] * norm1 * norm2, in[xzz] * norm1,
in[xxy] * norm1,         in[xyy] * norm1,         in[xyz] * norm1 * norm2,
in[xyy] * norm1,         in[yyy],                 in[yyz] * norm1,
in[xyz] * norm1 * norm2, in[yyz] * norm1,         in[yzz] * norm1,
in[xxz] * norm1,         in[xyz] * norm1 * norm2, in[xzz] * norm1,
in[xyz] * norm1 * norm2, in[yyz] * norm1,         in[yzz] * norm1,
in[xzz] * norm1,         in[yzz] * norm1,         in[zzz]
	};

	double full_out[27];
	rotate_t3(rotmat, full_in, full_out);

	/* Q-Chem order */
	out[0] = full_out[9 * 0 + 3 * 0 + 0];
	out[1] = full_out[9 * 0 + 3 * 0 + 1] / norm1;
	out[2] = full_out[9 * 0 + 3 * 1 + 1] / norm1;
	out[3] = full_out[9 * 1 + 3 * 1 + 1];
	out[4] = full_out[9 * 0 + 3 * 0 + 2] / norm1;
	out[5] = full_out[9 * 0 + 3 * 1 + 2] / norm1 / norm2;
	out[6] = full_out[9 * 1 + 3 * 1 + 2] / norm1;
	out[7] = full_out[9 * 0 + 3 * 2 + 2] / norm1;
	out[8] = full_out[9 * 1 + 3 * 2 + 2] / norm1;
	out[9] = full_out[9 * 2 + 3 * 2 + 2];
}

void
efp_update_xr(struct frag *frag, const mat_t *rotmat)
{
	/* move LMO centroids */
	for (int i = 0; i < frag->n_lmo; i++)
		move_pt(VEC(frag->x), rotmat, frag->lib->lmo_centroids + i,
				frag->lmo_centroids + i);

	/* rotate wavefunction */
	for (int lmo = 0; lmo < frag->n_lmo; lmo++) {
		double *in = frag->lib->xr_wf + lmo * frag->xr_wf_size;
		double *out = frag->xr_wf + lmo * frag->xr_wf_size;

		for (int i = 0, func = 0; frag->shells[i]; i++) {
			switch (frag->shells[i]) {
			case 'S':
				func++;
				break;
			case 'L':
				func++;
				/* fall through */
			case 'P':
				mat_vec(rotmat, (const vec_t *)(in + func),
						(vec_t *)(out + func));
				func += 3;
				break;
			case 'D':
				rotate_func_d(rotmat, in + func, out + func);
				func += 6;
				break;
			case 'F':
				rotate_func_f(rotmat, in + func, out + func);
				func += 10;
				break;
			}
		}
	}
}
