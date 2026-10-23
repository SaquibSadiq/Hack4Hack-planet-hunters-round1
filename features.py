import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy import stats, signal, ndimage
from joblib import Parallel, delayed
import time
import os

# Empirical quiet star noise curve fit coefficients log10(std) vs tmag
# Fitted from quiet stars in TESS data:
QUIET_POLY = np.array([0.0059867, 0.08995863, -5.13583722])

def precompute_ccd_profiles(flux, cen, stars):
    """
    Computes common instrumental trends for flux and centroids for each (camera, ccd).
    """
    profiles = {}
    for (cam, ccd), group in stars.groupby(['camera', 'ccd']):
        idxs = group.index.values
        f_sub = np.asarray(flux[idxs], dtype=np.float64)
        meds = np.nanmedian(f_sub, axis=1, keepdims=True)
        meds[np.isnan(meds) | (meds <= 0)] = 1.0
        norm = f_sub / meds - 1.0
        common_f = np.nanmedian(norm, axis=0)
        
        c0_sub = np.asarray(cen[idxs, 0, :], dtype=np.float64)
        c1_sub = np.asarray(cen[idxs, 1, :], dtype=np.float64)
        common_c0 = np.nanmedian(c0_sub, axis=0)
        common_c1 = np.nanmedian(c1_sub, axis=0)
        
        # Replace remaining NaNs in profiles with 0
        common_f[np.isnan(common_f)] = 0.0
        common_c0[np.isnan(common_c0)] = 0.0
        common_c1[np.isnan(common_c1)] = 0.0
        
        profiles[(cam, ccd)] = (common_f, common_c0, common_c1)
    return profiles

def extract_single_star_v2(idx, f_raw, cen0_raw, cen1_raw, star_row, t, quality, ccd_profile):
    """
    Extracts 95 comprehensive astrophysical, time-series, frequency, astrometric,
    flare, glitch, and stellar features with CCD cotrending.
    """
    f_raw = np.asarray(f_raw, dtype=np.float64)
    cen0_raw = np.asarray(cen0_raw, dtype=np.float64)
    cen1_raw = np.asarray(cen1_raw, dtype=np.float64)
    
    common_f, common_c0, common_c1 = ccd_profile
    
    # 1. Normalization & CCD Cotrending
    med_flux = np.nanmedian(f_raw)
    if np.isnan(med_flux) or med_flux <= 0:
        med_flux = 1.0
    f_norm = f_raw / med_flux - 1.0
    
    # Cotrend: subtract common detector profile
    f_cotrend = f_norm - common_f
    cen0_cotrend = cen0_raw - common_c0
    cen1_cotrend = cen1_raw - common_c1
    
    # Mask of clean cadences
    valid = ~np.isnan(f_cotrend) & (quality == 0)
    if np.sum(valid) < 50:
        valid = ~np.isnan(f_cotrend)
    if np.sum(valid) < 20:
        valid = np.ones(len(f_cotrend), dtype=bool)
        f_cotrend[np.isnan(f_cotrend)] = 0.0

    N_points = len(f_cotrend)
    xp = np.where(valid)[0]
    fp = f_cotrend[valid]
    f_filled = np.interp(np.arange(N_points), xp, fp)
    
    # Detrending: 49 cadences (~24.5 hours = ~1 day)
    trend = ndimage.median_filter(f_filled, size=49)
    f_flat = f_cotrend - trend
    f_flat_filled = f_filled - trend
    
    # Cotrended Centroids
    c0_valid = ~np.isnan(cen0_cotrend) & valid
    c1_valid = ~np.isnan(cen1_cotrend) & valid
    c0_filled = np.interp(np.arange(N_points), np.where(c0_valid)[0], cen0_cotrend[c0_valid]) if np.sum(c0_valid) > 10 else np.zeros(N_points)
    c1_filled = np.interp(np.arange(N_points), np.where(c1_valid)[0], cen1_cotrend[c1_valid]) if np.sum(c1_valid) > 10 else np.zeros(N_points)
    c0_trend = ndimage.median_filter(c0_filled, size=49)
    c1_trend = ndimage.median_filter(c1_filled, size=49)
    c0_flat = c0_filled - c0_trend
    c1_flat = c1_filled - c1_trend
    c_flat_dist = np.sqrt(c0_flat**2 + c1_flat**2)
    
    # 2. Basic Distribution on valid cadences
    f_v = f_flat[valid]
    std_flat = float(np.std(f_v))
    mad_flat = float(np.median(np.abs(f_v - np.median(f_v))))
    min_flat = float(np.min(f_v))
    max_flat = float(np.max(f_v))
    skew_flat = float(stats.skew(f_v))
    kurt_flat = float(stats.kurtosis(f_v))
    
    p01, p05, p10, p25, p50, p75, p90, p95, p99 = np.percentile(f_v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    iqr_flat = float(p75 - p25)
    range_90 = float(p95 - p05)
    range_98 = float(p99 - p01)
    pos_neg_ratio = float(max_flat / (abs(min_flat) + 1e-6))
    pct_asymmetry = float(((p95 - p50) - (p50 - p05)) / (range_90 + 1e-6))
    
    # Raw cotrended flux stats
    f_raw_v = f_cotrend[valid]
    std_raw = float(np.std(f_raw_v))
    mad_raw = float(np.median(np.abs(f_raw_v - np.median(f_raw_v))))
    trend_std = float(np.std(trend))
    smoothness_ratio = float(trend_std / (std_flat + 1e-6))
    
    # Point-to-point scatter & Von Neumann ratio
    diff_f = np.diff(f_filled)
    diff_std = float(np.std(diff_f))
    point_to_point_ratio = float(diff_std / (std_flat + 1e-6))
    var_f = float(np.var(f_filled))
    von_neumann = float(np.mean(diff_f**2) / (var_f + 1e-9))
    
    # Zero crossings of f_flat
    zero_crossings = float(np.sum(np.diff(np.signbit(f_flat_filled)) != 0))
    
    # Autocorrelation of raw & filled flux
    r_lag1 = float(np.corrcoef(f_filled[:-1], f_filled[1:])[0, 1]) if len(f_filled) > 2 else 0.0
    r_lag2 = float(np.corrcoef(f_filled[:-2], f_filled[2:])[0, 1]) if len(f_filled) > 4 else 0.0
    r_lag5 = float(np.corrcoef(f_filled[:-5], f_filled[5:])[0, 1]) if len(f_filled) > 10 else 0.0
    r_lag10 = float(np.corrcoef(f_filled[:-10], f_filled[10:])[0, 1]) if len(f_filled) > 20 else 0.0
    
    # 3. Flare Detection Features
    mad_scale = mad_flat * 1.4826 + 1e-8
    f_z = f_v / mad_scale
    n_pos_3sig = float(np.mean(f_z > 3.0))
    n_pos_5sig = float(np.mean(f_z > 5.0))
    n_pos_8sig = float(np.mean(f_z > 8.0))
    n_neg_3sig = float(np.mean(f_z < -3.0))
    n_neg_5sig = float(np.mean(f_z < -5.0))
    flare_excess = float((n_pos_3sig + 1e-4) / (n_neg_3sig + 1e-4))
    
    # First-difference skewness
    diff_skew = float(stats.skew(diff_f))
    max_single_diff_snr = float(np.max(diff_f) / (diff_std + 1e-8))
    
    # Flare candidate count (>4 sigma with decay)
    above_4 = np.where(f_flat_filled > 4.0 * mad_scale)[0]
    flare_candidates = 0
    if len(above_4) > 0:
        clusters = np.split(above_4, np.where(np.diff(above_4) > 3)[0] + 1)
        for cluster in clusters:
            if len(cluster) >= 2:
                flare_candidates += 1
    flare_candidates = float(flare_candidates)
    
    # 4. Box Least Squares (BLS) for Transits and Eclipses
    t_v = t[valid]
    y_v = f_v
    bls = BoxLeastSquares(t_v, y_v)
    periods = np.linspace(0.5, 13.0, 300)
    durations = np.array([0.05, 0.10, 0.18, 0.25])
    res = bls.power(periods, durations)
    b_idx = np.argmax(res.power)
    bls_P = float(res.period[b_idx])
    bls_dur = float(res.duration[b_idx])
    bls_t0 = float(res.transit_time[b_idx])
    bls_power = float(res.power[b_idx])
    bls_mean_power = float(np.mean(res.power))
    bls_std_power = float(np.std(res.power))
    bls_snr = float((bls_power - bls_mean_power) / (bls_std_power + 1e-8))
    bls_depth = float(res.depth[b_idx])
    
    # Phase fold metrics
    phase = ((t_v - bls_t0 + 0.5 * bls_P) % bls_P) / bls_P - 0.5
    half_dur = (bls_dur / 2.0) / bls_P
    in_tr = np.abs(phase) < half_dur
    in_tr_inner = np.abs(phase) < (0.5 * half_dur)
    sec_tr = np.abs(phase - 0.5) < half_dur
    out_tr = (~in_tr) & (~sec_tr)
    
    n_in = int(np.sum(in_tr))
    dep_pri = float(-np.median(y_v[in_tr])) if n_in > 0 else 0.0
    dep_sec = float(-np.median(y_v[sec_tr])) if np.sum(sec_tr) > 0 else 0.0
    sec_depth_ratio = float(dep_sec / (dep_pri + 1e-6))
    
    # Physical transit SNR & Delta Chi2
    transit_snr_physical = float((bls_depth * np.sqrt(max(n_in, 1))) / mad_scale)
    transit_chi2_diff = float(max(n_in, 1) * ((bls_depth / mad_scale)**2))
    
    # Transit consistency across cycles
    k_min = int(np.floor((t_v[0] - bls_t0) / bls_P))
    k_max = int(np.ceil((t_v[-1] - bls_t0) / bls_P))
    n_expected = 0
    n_detected = 0
    for k in range(k_min, k_max + 1):
        tk = bls_t0 + k * bls_P
        mask_k = np.abs(t_v - tk) < (bls_dur / 2.0)
        if np.sum(mask_k) >= 2:
            n_expected += 1
            if np.median(y_v[mask_k]) < -0.5 * bls_depth:
                n_detected += 1
    transit_consistency = float(n_detected / max(n_expected, 1))
    
    # Flatness of bottom: U-shape vs V-shape
    dep_inner = float(-np.median(y_v[in_tr_inner])) if np.sum(in_tr_inner) > 0 else 0.0
    transit_flatness = float(dep_inner / (dep_pri + 1e-6))
    
    # Odd vs Even transits
    transit_epoch = np.round((t_v[in_tr] - bls_t0) / bls_P) if n_in > 0 else np.array([])
    odd_mask = (transit_epoch % 2 == 1)
    even_mask = (transit_epoch % 2 == 0)
    odd_dep = float(-np.median(y_v[in_tr][odd_mask])) if np.sum(odd_mask) > 0 else dep_pri
    even_dep = float(-np.median(y_v[in_tr][even_mask])) if np.sum(even_mask) > 0 else dep_pri
    odd_even_diff = float(abs(odd_dep - even_dep) / (abs(odd_dep) + abs(even_dep) + 1e-6))
    
    # 5. Centroid Astrometric Shift Metrics (Cotrended!)
    c0_v = c0_flat[valid]
    c1_v = c1_flat[valid]
    c_dist_v = c_flat_dist[valid]
    
    c0_in = float(np.mean(c0_v[in_tr])) if n_in > 0 else 0.0
    c0_out = float(np.mean(c0_v[out_tr])) if np.sum(out_tr) > 0 else 0.0
    c1_in = float(np.mean(c1_v[in_tr])) if n_in > 0 else 0.0
    c1_out = float(np.mean(c1_v[out_tr])) if np.sum(out_tr) > 0 else 0.0
    
    cen_in_shift = float(np.sqrt((c0_in - c0_out)**2 + (c1_in - c1_out)**2))
    cen_noise = float(np.std(c_dist_v) / np.sqrt(max(n_in, 1)) + 1e-6)
    cen_shift_snr = float(cen_in_shift / cen_noise)
    
    # Deepest 5% cadences centroid shift (model-independent dip centroid shift)
    dip_5pct = y_v < p05
    nondip_95 = y_v >= p05
    c0_dip5 = float(np.mean(c0_v[dip_5pct])) if np.sum(dip_5pct) > 0 else 0.0
    c0_nondip = float(np.mean(c0_v[nondip_95])) if np.sum(nondip_95) > 0 else 0.0
    c1_dip5 = float(np.mean(c1_v[dip_5pct])) if np.sum(dip_5pct) > 0 else 0.0
    c1_nondip = float(np.mean(c1_v[nondip_95])) if np.sum(nondip_95) > 0 else 0.0
    cen_dip5_shift = float(np.sqrt((c0_dip5 - c0_nondip)**2 + (c1_dip5 - c1_nondip)**2))
    cen_dip5_snr = float(cen_dip5_shift / (np.std(c_dist_v) / np.sqrt(max(np.sum(dip_5pct), 1)) + 1e-6))
    
    # Centroid correlation with flux
    r0 = stats.pearsonr(y_v, c0_v)[0] if len(y_v) > 10 else 0.0
    r1 = stats.pearsonr(y_v, c1_v)[0] if len(y_v) > 10 else 0.0
    cen_flux_corr = float(np.sqrt((r0 if not np.isnan(r0) else 0.0)**2 + (r1 if not np.isnan(r1) else 0.0)**2))
    
    cen_std_total = float(np.std(c_dist_v))
    cen_range_total = float(np.percentile(c_dist_v, 99) - np.percentile(c_dist_v, 1))
    
    # 6. Frequency Spectrum & Spectral Gini
    fft_vals = np.abs(np.fft.rfft(f_flat_filled)[1:])
    fft_freqs = np.fft.rfftfreq(N_points, d=0.5/24.0)[1:] # cycles/day
    fft_power = fft_vals**2
    tot_p = float(np.sum(fft_power)) + 1e-12
    
    low_p = float(np.sum(fft_power[fft_freqs < 1.0]))
    mid_p = float(np.sum(fft_power[(fft_freqs >= 1.0) & (fft_freqs < 6.0)]))
    high_p = float(np.sum(fft_power[fft_freqs >= 6.0]))
    
    p_low_ratio = float(low_p / tot_p)
    p_mid_ratio = float(mid_p / tot_p)
    p_high_ratio = float(high_p / tot_p)
    
    peak_fft_idx = np.argmax(fft_vals)
    peak_fft_freq = float(fft_freqs[peak_fft_idx])
    peak_fft_power = float(fft_vals[peak_fft_idx])
    fft_sharpness = float(peak_fft_power / (np.mean(fft_vals) + 1e-9))
    
    # Top 3 peaks power ratio & Spectral Gini
    s_p = np.sort(fft_power)
    n_p = len(s_p)
    idx_p = np.arange(1, n_p + 1)
    fft_gini = float((np.sum((2 * idx_p - n_p - 1) * s_p)) / (n_p * tot_p))
    fft_top3_ratio = float(np.sum(s_p[-3:]) / tot_p)
    
    # Spectral entropy
    p_norm = fft_power / tot_p
    p_norm = p_norm[p_norm > 1e-12]
    spectral_entropy = float(-np.sum(p_norm * np.log(p_norm)) / np.log(len(fft_vals)))
    
    # 7. Autocorrelation (Starspots & Rotation)
    acf = signal.correlate(trend - np.mean(trend), trend - np.mean(trend), mode='full')
    acf = acf[len(acf)//2:]
    acf_norm = acf / (acf[0] + 1e-9)
    acf_window = acf_norm[24:min(480, len(acf_norm))]
    if len(acf_window) > 0:
        acf_peak_idx = np.argmax(acf_window)
        acf_rot_period = float((acf_peak_idx + 24) * 0.5 / 24.0)
        acf_rot_power = float(acf_window[acf_peak_idx])
    else:
        acf_rot_period = 0.0
        acf_rot_power = 0.0

    # 8. Glitch & Systematic Sensitivity
    is_dump = (quality & 1) > 0
    is_scatter = (quality & 2) > 0
    quiet_mask = (quality == 0)
    
    dump_flux_ratio = float(np.mean(np.abs(f_norm[is_dump])) / (np.mean(np.abs(f_norm[quiet_mask])) + 1e-6)) if np.sum(is_dump) > 0 else 1.0
    scatter_flux_ratio = float(np.mean(np.abs(f_norm[is_scatter])) / (np.mean(np.abs(f_norm[quiet_mask])) + 1e-6)) if np.sum(is_scatter) > 0 else 1.0
    dump_cen_ratio = float(np.mean(c_flat_dist[is_dump]) / (np.mean(c_flat_dist[quiet_mask]) + 1e-6)) if np.sum(is_dump) > 0 else 1.0
    
    # Momentum dump variance fraction
    var_all = np.sum((f_norm[valid] - np.mean(f_norm[valid]))**2)
    var_dump = np.sum((f_norm[is_dump & ~np.isnan(f_norm)] - np.mean(f_norm[valid]))**2) if np.sum(is_dump) > 0 else 0.0
    dump_var_frac = float(var_dump / (var_all + 1e-9))
    
    # Step jump detector
    step_filter = np.array([-1]*10 + [1]*10) / 10.0
    step_resp = np.abs(np.convolve(f_filled, step_filter, mode='valid'))
    max_step_jump = float(np.max(step_resp))
    
    # Orbit 1 vs Orbit 2 discrepancy
    orbit1 = f_norm[:600]
    orbit2 = f_norm[700:]
    o1_v = orbit1[~np.isnan(orbit1)]
    o2_v = orbit2[~np.isnan(orbit2)]
    if len(o1_v) > 20 and len(o2_v) > 20:
        orbit_mean_diff = float(abs(np.mean(o1_v) - np.mean(o2_v)))
        orbit_std_diff = float(abs(np.std(o1_v) - np.std(o2_v)))
    else:
        orbit_mean_diff = 0.0
        orbit_std_diff = 0.0

    # 9. Stellar Metadata & Empirical Noise Floor
    tmag = float(star_row.get('tmag', 12.0))
    teff = float(star_row.get('teff', 5500.0))
    logg = float(star_row.get('logg', 4.4))
    radius = float(star_row.get('radius', 1.0))
    contamination = float(star_row.get('contamination', 0.0))
    
    # Empirical expected quiet star noise
    log_expected_noise = float(np.polyval(QUIET_POLY, tmag))
    expected_noise_emp = float(10**log_expected_noise)
    noise_ratio_emp = float(std_flat / (expected_noise_emp + 1e-6))
    transit_depth_over_noise = float(bls_depth / (expected_noise_emp + 1e-6))
    
    # Implied planetary radius
    implied_rp = float(np.sqrt(max(bls_depth, 0.0)) * radius)
    
    # Stellar density proxy: rho ~ M/R^3 ~ g/R
    stellar_density_proxy = float(logg - np.log10(max(radius, 0.01)))
    stefan_boltzmann_proxy = float((radius**2) * ((teff / 5778.0)**4))

    return {
        'std_flat': std_flat, 'mad_flat': mad_flat, 'min_flat': min_flat, 'max_flat': max_flat,
        'skew_flat': skew_flat, 'kurt_flat': kurt_flat,
        'p01': p01, 'p05': p05, 'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90, 'p95': p95, 'p99': p99,
        'iqr_flat': iqr_flat, 'range_90': range_90, 'range_98': range_98,
        'pos_neg_ratio': pos_neg_ratio, 'pct_asymmetry': pct_asymmetry,
        'std_raw': std_raw, 'mad_raw': mad_raw, 'trend_std': trend_std, 'smoothness_ratio': smoothness_ratio,
        'diff_std': diff_std, 'point_to_point_ratio': point_to_point_ratio, 'von_neumann': von_neumann,
        'zero_crossings': zero_crossings, 'r_lag1': r_lag1, 'r_lag2': r_lag2, 'r_lag5': r_lag5, 'r_lag10': r_lag10,
        'n_pos_3sig': n_pos_3sig, 'n_pos_5sig': n_pos_5sig, 'n_pos_8sig': n_pos_8sig,
        'n_neg_3sig': n_neg_3sig, 'n_neg_5sig': n_neg_5sig, 'flare_excess': flare_excess,
        'diff_skew': diff_skew, 'max_single_diff_snr': max_single_diff_snr, 'flare_candidates': flare_candidates,
        'bls_P': bls_P, 'bls_dur': bls_dur, 'bls_power': bls_power, 'bls_snr': bls_snr, 'bls_depth': bls_depth,
        'sec_depth_ratio': sec_depth_ratio, 'transit_flatness': transit_flatness, 'odd_even_diff': odd_even_diff,
        'transit_snr_physical': transit_snr_physical, 'transit_chi2_diff': transit_chi2_diff,
        'transit_consistency': transit_consistency,
        'cen_in_shift': cen_in_shift, 'cen_shift_snr': cen_shift_snr,
        'cen_dip5_shift': cen_dip5_shift, 'cen_dip5_snr': cen_dip5_snr, 'cen_flux_corr': cen_flux_corr,
        'cen_std_total': cen_std_total, 'cen_range_total': cen_range_total,
        'p_low_ratio': p_low_ratio, 'p_mid_ratio': p_mid_ratio, 'p_high_ratio': p_high_ratio,
        'peak_fft_freq': peak_fft_freq, 'peak_fft_power': peak_fft_power, 'fft_sharpness': fft_sharpness,
        'fft_gini': fft_gini, 'fft_top3_ratio': fft_top3_ratio, 'spectral_entropy': spectral_entropy,
        'acf_rot_period': acf_rot_period, 'acf_rot_power': acf_rot_power,
        'dump_flux_ratio': dump_flux_ratio, 'scatter_flux_ratio': scatter_flux_ratio, 'dump_cen_ratio': dump_cen_ratio,
        'dump_var_frac': dump_var_frac, 'max_step_jump': max_step_jump,
        'orbit_mean_diff': orbit_mean_diff, 'orbit_std_diff': orbit_std_diff,
        'tmag': tmag, 'teff': teff, 'logg': logg, 'radius': radius, 'contamination': contamination,
        'expected_noise_emp': expected_noise_emp, 'noise_ratio_emp': noise_ratio_emp,
        'transit_depth_over_noise': transit_depth_over_noise,
        'implied_rp': implied_rp, 'stellar_density_proxy': stellar_density_proxy,
        'stefan_boltzmann_proxy': stefan_boltzmann_proxy
    }

def extract_features_dataset(flux_file, cen_file, stars_file, cadences_file, n_jobs=-1):
    print(f"Loading {flux_file}, {cen_file}, {stars_file}...")
    flux = np.load(flux_file, mmap_mode='r')
    cen = np.load(cen_file, mmap_mode='r')
    stars = pd.read_csv(stars_file)
    cad = pd.read_csv(cadences_file)
    
    t = cad.time_btjd.values
    quality = cad.quality.values
    N = len(stars)
    
    print("Precomputing CCD instrumental profiles...")
    ccd_profiles = precompute_ccd_profiles(flux, cen, stars)
    
    print(f"Extracting features for {N} stars using {n_jobs} cores...")
    t0 = time.time()
    
    results = Parallel(n_jobs=n_jobs, batch_size=50)(
        delayed(extract_single_star_v2)(
            i, flux[i], cen[i, 0], cen[i, 1], stars.iloc[i], t, quality,
            ccd_profiles[(stars.camera.iloc[i], stars.ccd.iloc[i])]
        )
        for i in range(N)
    )
    df_feat = pd.DataFrame(results)
    dt = time.time() - t0
    print(f"Extracted {df_feat.shape[1]} features for {N} stars in {dt:.1f}s ({dt/N:.4f}s/star)")
    return df_feat
