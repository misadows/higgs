import numpy as np
from particle import Particle
from math_utils import * 


class RhoRhoEvent(object):
    def __init__(self, data, args, debug=True):
        # [n, pi-, pi0, an, pi+, pi0]

        p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(6)]
        cols = []

        def get_tau1(p):
            tau1_nu = p[0]
            tau1_pi = p[1:3]
            tau1_rho = tau1_pi[0] + tau1_pi[1]

            return tau1_nu, tau1_pi, tau1_rho

        def get_tau2(p):
            tau2_nu = p[3]
            tau2_pi = p[4:6]
            tau2_rho = tau2_pi[0] + tau2_pi[1]

            return tau2_nu, tau2_pi, tau2_rho

        p_tau1_nu, l_tau1_pi, p_tau1_rho = get_tau1(p) # p- particle, l-list
        p_tau2_nu, l_tau2_pi, p_tau2_rho = get_tau2(p)

        rho_rho = p_tau1_rho + p_tau2_rho

        PHI, THETA = calc_angles(p_tau1_rho, rho_rho)
        lambda_noise = args.LAMBDA

        # all particles boosted & rotated
        for i, idx in enumerate([0, 1, 2, 3, 4, 5]):
            part = boost_and_rotate(p[idx], PHI, THETA, rho_rho)
            if args.FEAT in ["Model-OnlyHad", "Model-Benchmark", "Model-1", "Model-2", "Model-3"]:
                if idx not in [0, 3]:
                    cols.append(part.vec)
            if args.FEAT == "Model-Oracle":
                cols.append(part.vec)

        # rho particles & recalculated mass 
        if args.FEAT == "Model-OnlyHad":
            for i, rho in enumerate([p_tau1_rho] + [p_tau2_rho]):
                rho = boost_and_rotate(rho, PHI, THETA, rho_rho)
                cols.append(rho.vec)
                cols.append(rho.recalculated_mass)

        if args.FEAT == "Model-OnlyHad":
            cols += [get_acoplanar_angle(p[1], p[2], p[4], p[5], rho_rho)]
            cols += [get_y(p[1], p[2], rho_rho), get_y(p[4], p[5], rho_rho)]

        #------------------------------------------------------------

        pb_tau1_h = boost_and_rotate(p_tau1_rho, PHI, THETA, rho_rho)
        pb_tau2_h = boost_and_rotate(p_tau2_rho, PHI, THETA, rho_rho)
        pb_tau1_nu = boost_and_rotate(p_tau1_nu, PHI, THETA, rho_rho)
        pb_tau2_nu = boost_and_rotate(p_tau2_nu, PHI, THETA, rho_rho)

        #------------------------------------------------------------

        v_ETmiss_x = p_tau1_nu.x + p_tau2_nu.x
        v_ETmiss_y = p_tau1_nu.y + p_tau2_nu.y
        if args.FEAT == "Model-2":
            cols += [v_ETmiss_x, v_ETmiss_y]

        #------------------------------------------------------------

        if args.METHOD == "A":
            va_alpha1, va_alpha2 = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "B":
            va_alpha1, va_alpha2 = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "C":
            va_alpha1, va_alpha2 = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)

        #------------------------------------------------------------

        va_tau1_nu_long = va_alpha1 * pb_tau1_h.z
        va_tau2_nu_long = va_alpha2 * pb_tau2_h.z

        va_tau1_nu_E = approx_E_nu(pb_tau1_h, va_tau1_nu_long)
        va_tau2_nu_E = approx_E_nu(pb_tau2_h, va_tau2_nu_long)

        #------------------------------------------------------------

        va_tau1_nu_trans = np.sqrt(np.square(va_tau1_nu_E) - np.square(va_tau1_nu_long))
        va_tau2_nu_trans = np.sqrt(np.square(va_tau2_nu_E) - np.square(va_tau2_nu_long))

        for i, elem1 in enumerate(va_tau1_nu_trans):
            if (np.isnan(elem1)):
                va_tau1_nu_trans[i] = 0

        for i, elem1 in enumerate(va_tau2_nu_trans):
            if (np.isnan(elem1)):
                va_tau2_nu_trans[i] = 0   

        v_tau1_nu_phi = np.arctan2(pb_tau1_nu.x, pb_tau1_nu.y) #boosted
        v_tau2_nu_phi = np.arctan2(pb_tau2_nu.x, pb_tau2_nu.y)
        vn_tau1_nu_phi = smear_exp(v_tau1_nu_phi, lambda_noise)
        vn_tau2_nu_phi = smear_exp(v_tau2_nu_phi, lambda_noise)

        tau1_sin_phi = np.sin(vn_tau1_nu_phi)
        tau1_cos_phi = np.cos(vn_tau1_nu_phi)
        tau2_sin_phi = np.sin(vn_tau2_nu_phi)
        tau2_cos_phi = np.cos(vn_tau2_nu_phi)

        #------------------------------------------------------------

        if args.FEAT in ["Model-1", "Model-2"]:
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans, va_tau2_nu_trans]
        elif args.FEAT == "Model-3":
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans * tau1_sin_phi, va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]
        elif args.FEAT == "Model-Benchmark":
            cols += [va_tau1_nu_trans * tau1_sin_phi, va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]

        # filter
        filt = (p_tau1_rho.pt >= 20)
        for part in (l_tau1_pi + l_tau2_pi):
            filt = filt & (part.pt >= 1)
        filt = filt.astype(np.float32)

        if args.FEAT in ["Model-Oracle", "Model-OnlyHad"]:
            cols += [filt]

        elif args.FEAT in ["Model-Benchmark", "Model-1", "Model-2", "Model-3"]:
            isFilter = np.full(rho_rho.e.shape, True, dtype=bool)
            for alpha in [va_alpha1, va_alpha2]:
                isFilter *= (alpha > 0)
            for energy in [va_tau1_nu_E, va_tau2_nu_E]:
                isFilter *= (energy > 0)
            for trans_comp in [va_tau1_nu_trans, va_tau2_nu_trans]:
                isFilter *= np.logical_not(np.isnan(trans_comp))
            cols += [filt * isFilter]

        for i in range(len(cols)):
            if len(cols[i].shape) == 1:
                cols[i] = cols[i].reshape([-1, 1])

        self.cols = np.concatenate(cols, 1)
