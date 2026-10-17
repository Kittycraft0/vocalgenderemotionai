"""
Dynamic Confidence & Reliability Engine for Vocal Analysis and Sound Reconstruction
Author / Pair: Antigravity & User
Workspace: vocalgenderemotionai
File: dynamic_confidence.py

Predicts when analysis frames are accurate vs inaccurate in real time.
Instead of a static weight, this provides dynamic, frame-by-frame confidence scores [0.0 to 1.0]
derived from acoustic physics, vocal tract biomechanics, and benchmark error patterns:
  - Energy & SNR gating (distinguishes speech from room noise / hiss)
  - Zero-crossing rate & fricative mismatch penalties
  - Formant bandwidth physiological sanity bounds (30 - 350 Hz)
  - Pole collision and order consistency checks (F1 < F2 < F3 < F4)
  - Pitch temporal continuity & octave jump detection
  - Spectral dynamic range & harmonic prominence
"""

import math
import numpy as np


class DynamicConfidenceEngine:
    """
    Computes real-time dynamic confidence scores for pitch, formants,
    vocal weight, and harmonics.
    """

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        # Physiological formant ranges for human vocal tracts (Hz)
        self.formant_bounds = {
            1: (180.0, 1150.0),    # F1: close vowels [i, u] to open [a]
            2: (550.0, 2900.0),    # F2: back vowels [u, o] to front [i]
            3: (1300.0, 3900.0),   # F3: rhoticity / tract length
            4: (2200.0, 4800.0),   # F4: anatomical length
            5: (3200.0, 5500.0)    # F5: unverified / high variance
        }

    def compute_pitch_confidence(self, pitch, prev_pitch=None, rms_energy=None, zcr=None, harm_mags=None):
        """
        Computes dynamic confidence in fundamental frequency F0:
        Returns confidence score in [0.0, 1.0].
        """
        if pitch is None or pitch <= 0.0 or math.isnan(pitch):
            return 0.0

        conf = 1.0

        # 1. Human vocal pitch physiological plausibility (55 Hz to 900 Hz for speech)
        if pitch < 55.0 or pitch > 950.0:
            conf *= 0.30
        elif pitch < 70.0 or pitch > 650.0:
            conf *= 0.70

        # 2. Energy / SNR check (if available)
        if rms_energy is not None:
            if rms_energy < 0.0015:
                # Very low energy near mic noise floor: high probability of false alarm
                conf *= 0.15
            elif rms_energy < 0.005:
                conf *= 0.60

        # 3. Zero-Crossing Rate check (consonant / fricative detection)
        # Voiced vowels have low ZCR (<0.10); unvoiced fricatives like /s/ have ZCR > 0.18
        if zcr is not None:
            if zcr > 0.22:
                conf *= 0.10  # Plosive/fricative noise: pitch is an artifact
            elif zcr > 0.15:
                conf *= 0.50

        # 4. Harmonic prominence check (FFT peak emergence)
        if harm_mags is not None and len(harm_mags) > 0:
            h1_mag = harm_mags[0]
            if h1_mag < -65.0:
                # Fundamental is buried in noise floor
                conf *= 0.25
            elif h1_mag < -50.0:
                conf *= 0.75

        # 5. Temporal pitch continuity / Octave jump penalty
        if prev_pitch is not None and prev_pitch > 40.0:
            semitone_diff = 12.0 * abs(math.log2(pitch / prev_pitch))
            if semitone_diff > 10.0:
                # Sudden >10 semitone leap in a single 30ms frame: likely octave doubling/halving
                conf *= 0.40
            elif semitone_diff > 5.0:
                conf *= 0.80

        return float(np.clip(conf, 0.0, 1.0))

    def compute_formant_confidences(self, formants, bandwidths, intensities=None,
                                    pitch_conf=1.0, prev_formants=None):
        """
        Computes individual dynamic confidence scores for F1..F5:
        Returns list of floats [c_F1, c_F2, c_F3, c_F4, c_F5] each in [0.0, 1.0].
        """
        num_f = min(len(formants), 5)
        confidences = [0.0] * 5

        for k in range(num_f):
            fk = formants[k] if k < len(formants) else 0.0
            bk = bandwidths[k] if k < len(bandwidths) else 0.0
            ik = intensities[k] if (intensities is not None and k < len(intensities)) else 0.0

            if fk <= 50.0 or math.isnan(fk):
                confidences[k] = 0.0
                continue

            # F5 is not empirically verified in human speech datasets
            if k == 4:
                confidences[k] = 0.0
                continue

            c = 1.0
            formant_num = k + 1

            # 1. Physiological frequency bounds
            lo, hi = self.formant_bounds.get(formant_num, (100.0, 5000.0))
            if fk < lo or fk > hi:
                c *= 0.35

            # 2. Bandwidth physiological sanity check
            # Real human vocal tract bandwidths are 35 Hz to 280 Hz.
            # Bandwidths > 450 Hz or < 15 Hz are classic LPC pole merging/splitting artifacts.
            if bk > 650.0 or bk < 12.0:
                c *= 0.15
            elif bk > 400.0 or bk < 25.0:
                c *= 0.55
            elif 40.0 <= bk <= 250.0:
                c *= 1.05  # Ideal human resonance bandwidth

            # 3. Formant order / pole collision check
            # F(k) must be strictly greater than F(k-1) by at least 80 Hz
            if k > 0 and formants[k - 1] > 50.0:
                spacing = fk - formants[k - 1]
                if spacing < 75.0:
                    # Pole collision / two poles modeling the same resonance
                    c *= 0.20
                elif spacing < 140.0:
                    c *= 0.65

            # 4. Spectral intensity prominence over noise floor
            if intensities is not None and len(intensities) > k:
                if ik < -85.0:
                    c *= 0.20
                elif ik < -65.0:
                    c *= 0.65

            # 5. Voicing / Pitch confidence coupling
            # When unvoiced, LPC picks up ambient room transfer function rather than vocal tract
            if pitch_conf < 0.25:
                c *= 0.30
            elif pitch_conf < 0.60:
                c *= 0.70

            # 6. Temporal continuity
            if prev_formants is not None and k < len(prev_formants) and prev_formants[k] > 50.0:
                delta_hz = abs(fk - prev_formants[k])
                if delta_hz > 500.0:
                    # Giant 500 Hz jump in 30ms: pole swap
                    c *= 0.50
                elif delta_hz > 250.0:
                    c *= 0.80

            confidences[k] = float(np.clip(c, 0.0, 1.0))

        return confidences

    def compute_frame_confidence(self, rec, prev_rec=None, static_weights=None):
        """
        Analyzes a single frame record (from JSONL log or live analysis)
        and computes dynamic confidence scores for all acoustic features.

        Returns dictionary of dynamic weights [0.0 to 1.0] ready for synthesis.
        """
        static_weights = static_weights or {
            "pitch": 0.95, "F1": 1.00, "F2": 0.92, "F3": 0.72,
            "F4": 0.25, "F5": 0.00, "vocalweight": 0.77,
            "harmonic_series_lower": 0.90, "harmonic_series_upper": 0.20
        }

        p = rec.get("pitch", 0.0)
        prev_p = prev_rec.get("pitch", 0.0) if prev_rec else None
        harm_mags = rec.get("harmonic_mags", [])

        # 1. Pitch dynamic confidence
        c_pitch = self.compute_pitch_confidence(
            pitch=p,
            prev_pitch=prev_p,
            harm_mags=harm_mags
        )

        # 2. Formant dynamic confidences
        fmts = rec.get("formants", [])
        bws = rec.get("formant_bandwidths", [])
        ints = rec.get("formant_intensities", None)
        prev_fmts = prev_rec.get("formants", None) if prev_rec else None

        c_fmts = self.compute_formant_confidences(
            formants=fmts,
            bandwidths=bws,
            intensities=ints,
            pitch_conf=c_pitch,
            prev_formants=prev_fmts
        )

        # 3. Vocal weight confidence: reliable when voiced and energy is present
        w = rec.get("vocalweight", 0.0)
        c_vw = c_pitch * (1.0 if 5.0 <= w <= 95.0 else 0.4)

        # 4. Harmonic series confidence
        c_harm_lower = c_pitch * (0.95 if (harm_mags and harm_mags[0] > -60.0) else 0.40)
        c_harm_upper = c_harm_lower * 0.35

        # Multiply static benchmark calibration by dynamic confidence
        dynamic_weights = {
            "pitch_conf": c_pitch,
            "pitch_weight": static_weights.get("pitch", 0.95) * c_pitch,
            "F1_conf": c_fmts[0],
            "F1_weight": static_weights.get("F1", 1.00) * c_fmts[0],
            "F2_conf": c_fmts[1],
            "F2_weight": static_weights.get("F2", 0.92) * c_fmts[1],
            "F3_conf": c_fmts[2],
            "F3_weight": static_weights.get("F3", 0.72) * c_fmts[2],
            "F4_conf": c_fmts[3],
            "F4_weight": static_weights.get("F4", 0.25) * c_fmts[3],
            "F5_conf": 0.0,
            "F5_weight": 0.0,
            "vocalweight_conf": c_vw,
            "vocalweight_weight": static_weights.get("vocalweight", 0.77) * c_vw,
            "harmonic_lower_weight": static_weights.get("harmonic_series_lower", 0.90) * c_harm_lower,
            "harmonic_upper_weight": static_weights.get("harmonic_series_upper", 0.20) * c_harm_upper,
            "overall_speech_confidence": float(np.mean([c_pitch, c_fmts[0], c_fmts[1]]))
        }

        return dynamic_weights

