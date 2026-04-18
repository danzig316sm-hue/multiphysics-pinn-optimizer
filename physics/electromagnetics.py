"""
Magnetostatic physics solver module for electromagnetic analysis.

Handles electromagnetic physics for 15-kW Bergey direct-drive radial-flux outer-rotor PMSG
(60-slot / 50-pole, 150 rpm, N48H + BAAM composite) based on NREL MADE3D research.

Components:
- EMConstants: Electromagnetic material and geometry constants
- MagneticCircuitModel: Analytical magnetic circuit with reluctance network
- IronLossModel: Steinmetz iron loss and copper loss computation
- DemagnetizationChecker: Temperature-dependent demag safety analysis
- EMPINNLoss: Differentiable EM physics constraints for PINN training
"""

import torch
import numpy as np
import math


class EMConstants:
    """
    Electromagnetic constants for N48H + BAAM composite PMSG.

    Based on NREL MADE3D specification for 15-kW Bergey direct-drive generator.
    """

    # Material properties - N48H sintered neodymium magnets
    BR_N48H = 1.37  # Tesla, remanence at 20°C
    MU_R_N48H = 1.05  # relative permeability
    DEMAG_THRESHOLD_N48H = 0.45  # Tesla, at 60°C
    TEMP_COEFF_N48H = -0.12  # %/°C, temperature coefficient of Br

    # Material properties - BAAM printed composite magnets
    BR_BAAM = 0.87  # Tesla, 20 MGOe composite
    MU_R_BAAM = 1.10  # relative permeability
    DEMAG_THRESHOLD_BAAM = 0.35  # Tesla
    TEMP_COEFF_BAAM = -0.15  # %/°C

    # M-15 steel core saturation
    B_SAT_M15 = 1.8  # Tesla, saturation flux density

    # Machine geometry
    N_POLES = 50  # pole pairs
    N_SLOTS = 60  # stator slots
    RPM_RATED = 150  # rated mechanical speed
    MU_0 = 4.0 * np.pi * 1e-7  # H/m, vacuum permeability

    # Dimensions
    AIR_GAP_MM = 3.0  # mm, mechanical air gap
    AIR_GAP = AIR_GAP_MM * 1e-3  # m
    STACK_LENGTH = 350e-3  # m, active stack length

    # Approximate rotor outer radius (from NREL MADE3D 15kW spec)
    ROTOR_RADIUS = 0.42  # m, outer rotor radius

    # Pole arc ratio (typical for PMSG: 0.75-0.85)
    POLE_ARC_RATIO = 0.80  # fraction of pole pitch

    # Winding
    N_PHASES = 3  # three-phase
    TURNS_PER_PHASE = 60  # approximate turns per phase coil

    # Thermal
    REFERENCE_TEMP = 20.0  # °C, reference temperature for Br

    @classmethod
    def pole_pitch(cls):
        """Geometric pole pitch in radians."""
        return 2.0 * np.pi / cls.N_POLES

    @classmethod
    def slot_pitch(cls):
        """Geometric slot pitch in radians."""
        return 2.0 * np.pi / cls.N_SLOTS

    @classmethod
    def pole_arc_length(cls):
        """Arc length covered by one pole face."""
        return cls.POLE_ARC_RATIO * cls.pole_pitch() * cls.ROTOR_RADIUS


class MagneticCircuitModel:
    """
    Analytical magnetic circuit model using reluctance network.

    Computes air gap flux density, back-EMF, cogging torque, and output torque
    based on standard PMSG analytical design equations.
    """

    def __init__(self, br=EMConstants.BR_N48H, mu_r=EMConstants.MU_R_N48H,
                 hm=None, g=None, stack_length=None, rotor_radius=None):
        """
        Initialize magnetic circuit model.

        Args:
            br (float): Remanence in Tesla
            mu_r (float): Relative permeability
            hm (float): Magnet thickness in m (if None, computed)
            g (float): Air gap in m (if None, uses constant)
            stack_length (float): Stack length in m (if None, uses constant)
            rotor_radius (float): Rotor radius in m (if None, uses constant)
        """
        self.br = br
        self.mu_r = mu_r
        self.hm = hm if hm is not None else 0.012  # 12 mm default
        self.g = g if g is not None else EMConstants.AIR_GAP
        self.stack_length = stack_length if stack_length is not None else EMConstants.STACK_LENGTH
        self.rotor_radius = rotor_radius if rotor_radius is not None else EMConstants.ROTOR_RADIUS
        self.mu_0 = EMConstants.MU_0

    def compute_airgap_flux_density(self):
        """
        Compute air gap flux density using reluctance network.

        Simplified magnetic circuit with magnet, air gap, and stator/rotor iron.
        Iron permeability assumed very high (reluctance neglected).

        Returns:
            float: Air gap flux density in Tesla
        """
        # Magnet reluctance (with permeability)
        A_mag = self.stack_length * EMConstants.ROTOR_RADIUS * EMConstants.POLE_ARC_RATIO
        R_mag = self.hm / (self.mu_0 * self.mu_r * A_mag)

        # Air gap reluctance (two sides for symmetry: rotor and stator)
        R_gap = (2.0 * self.g) / (self.mu_0 * A_mag)

        # Total reluctance
        R_total = R_mag + R_gap

        # MMF from magnet
        H_mag = self.br / (self.mu_0 * self.mu_r)
        MMF = H_mag * self.hm

        # Air gap flux and flux density
        flux = MMF / R_total
        B_g = flux / A_mag

        return B_g

    def compute_flux_per_pole(self, B_g=None):
        """
        Compute magnetic flux per pole.

        Args:
            B_g (float): Air gap flux density in Tesla (if None, computed)

        Returns:
            float: Flux per pole in Weber
        """
        if B_g is None:
            B_g = self.compute_airgap_flux_density()

        # Pole face area
        pole_arc_length = EMConstants.POLE_ARC_RATIO * EMConstants.pole_pitch() * self.rotor_radius
        A_pole = pole_arc_length * self.stack_length

        flux = B_g * A_pole
        return flux

    def compute_back_emf(self, flux=None, n_turns=None, rpm=None, n_poles=None):
        """
        Compute RMS back-EMF.

        For sinusoidal flux distribution:
        E_rms = (√2 / π) × k_w × n_poles × flux × rpm / 60

        Args:
            flux (float): Flux per pole in Weber (if None, computed)
            n_turns (int): Turns per phase (if None, uses constant)
            rpm (float): Mechanical speed in rpm (if None, uses rated)
            n_poles (int): Number of pole pairs (if None, uses constant)

        Returns:
            float: RMS back-EMF in Volts
        """
        if flux is None:
            flux = self.compute_flux_per_pole()
        if n_turns is None:
            n_turns = EMConstants.TURNS_PER_PHASE
        if rpm is None:
            rpm = EMConstants.RPM_RATED
        if n_poles is None:
            n_poles = EMConstants.N_POLES

        # Winding factor (for distributed 3-phase winding, typical k_w ≈ 0.96)
        k_w = 0.96

        # Back-EMF constant: peak flux linkage change per mechanical revolution
        E_peak = np.sqrt(2.0) / np.pi * k_w * n_poles * flux * rpm / 60.0

        # RMS value for sinusoidal waveform
        E_rms = E_peak / np.sqrt(2.0)

        return E_rms

    def compute_cogging_torque_peak(self, B_g=None, n_slots=None, n_poles=None):
        """
        Compute peak cogging torque.

        Based on simplified formula for 50-pole/60-slot configuration:
        T_cog = (3/2) × B_g² × stack_length × rotor_radius² × pole_arc × lcm_factor / μ₀

        Args:
            B_g (float): Air gap flux density in Tesla (if None, computed)
            n_slots (int): Number of slots (if None, uses constant)
            n_poles (int): Number of pole pairs (if None, uses constant)

        Returns:
            float: Peak cogging torque in N⋅m
        """
        if B_g is None:
            B_g = self.compute_airgap_flux_density()
        if n_slots is None:
            n_slots = EMConstants.N_SLOTS
        if n_poles is None:
            n_poles = EMConstants.N_POLES

        # Slot opening factor (typical: 0.1-0.15 of slot pitch)
        slot_opening = 0.12 * EMConstants.slot_pitch() * self.rotor_radius

        # Pole arc
        pole_arc = EMConstants.POLE_ARC_RATIO * EMConstants.pole_pitch() * self.rotor_radius

        # LCM of slots and poles for slot-pole interaction
        lcm_sp = (n_slots * n_poles) // math.gcd(n_slots, n_poles)

        # Cogging torque amplitude (empirical formula)
        T_cog = (3.0 / 2.0) * (B_g ** 2) * self.stack_length * (self.rotor_radius ** 2) \
                * pole_arc * slot_opening / self.mu_0 / lcm_sp

        return T_cog

    def compute_torque(self, current, B_g=None, n_turns=None, n_poles=None):
        """
        Compute average electromagnetic torque from Lorentz force.

        T = (3/2) × n_poles × Φ × I × k_w

        Args:
            current (float): RMS phase current in Amperes
            B_g (float): Air gap flux density in Tesla (if None, computed)
            n_turns (int): Turns per phase (if None, uses constant)
            n_poles (int): Number of pole pairs (if None, uses constant)

        Returns:
            float: Average electromagnetic torque in N⋅m
        """
        if B_g is None:
            B_g = self.compute_airgap_flux_density()
        if n_turns is None:
            n_turns = EMConstants.TURNS_PER_PHASE
        if n_poles is None:
            n_poles = EMConstants.N_POLES

        flux = self.compute_flux_per_pole(B_g)

        # Winding factor
        k_w = 0.96

        # Torque constant: T = k_t × I
        # k_t = (3/2) × n_poles × flux × k_w
        k_t = 1.5 * n_poles * flux * k_w

        torque = k_t * current
        return torque

    def compute_back_emf_thd(self, pole_arc_ratio=None):
        """
        Compute back-EMF THD from analytical Fourier series.

        For radial-flux PMSG with sinusoidal magnet distribution,
        THD is dominated by 5th and 7th harmonics from slot effects.

        Args:
            pole_arc_ratio (float): Pole arc / pole pitch ratio (if None, uses constant)

        Returns:
            float: Total Harmonic Distortion in percent
        """
        if pole_arc_ratio is None:
            pole_arc_ratio = EMConstants.POLE_ARC_RATIO

        # Fundamental amplitude (normalized to 1.0)
        V1 = 1.0

        # 5th harmonic (slot ripple)
        V5 = 0.08 * (1.0 - pole_arc_ratio)  # increases with slot opening

        # 7th harmonic
        V7 = 0.04 * (1.0 - pole_arc_ratio)

        # 11th and 13th harmonics (smaller)
        V11 = 0.02 * (1.0 - pole_arc_ratio)
        V13 = 0.01 * (1.0 - pole_arc_ratio)

        # THD calculation
        thd = 100.0 * np.sqrt(V5**2 + V7**2 + V11**2 + V13**2) / V1

        return thd

    def compute_efficiency(self, torque, rpm, copper_loss, iron_loss,
                          mechanical_loss=None):
        """
        Compute generator efficiency.

        η = P_out / (P_out + P_losses)

        Args:
            torque (float): Electromagnetic torque in N⋅m
            rpm (float): Mechanical speed in rpm
            copper_loss (float): Copper loss in Watts
            iron_loss (float): Iron loss in Watts
            mechanical_loss (float): Mechanical loss in Watts (if None, 2% of output)

        Returns:
            float: Efficiency in percent
        """
        # Mechanical power
        omega = 2.0 * np.pi * rpm / 60.0  # rad/s
        P_mech = torque * omega

        if mechanical_loss is None:
            mechanical_loss = 0.02 * P_mech

        # Total losses
        P_total_loss = copper_loss + iron_loss + mechanical_loss

        # Output power (generator)
        P_out = P_mech - P_total_loss

        # Efficiency
        if P_mech > 0:
            eta = 100.0 * P_out / P_mech
        else:
            eta = 0.0

        return eta


class IronLossModel:
    """
    Iron loss and copper loss computation using Steinmetz equation.

    Includes hysteresis and eddy current components.
    """

    # Steinmetz coefficients for M-15 steel at 1 kHz
    K_H = 0.019  # hysteresis loss coefficient (W/kg/(T^1.8))
    K_E = 0.00012  # eddy current loss coefficient (W/kg/(T^2))
    ALPHA = 1.8  # hysteresis exponent
    BETA = 2.0  # eddy current exponent

    @staticmethod
    def compute_iron_loss(B_peak, frequency, mass_core,
                         k_h=K_H, k_e=K_E, alpha=ALPHA, beta=BETA):
        """
        Compute iron core losses using Steinmetz equation.

        P_iron = P_h + P_e
        P_h = k_h × f × B_peak^α  (hysteresis)
        P_e = k_e × f² × B_peak^β  (eddy current)

        Args:
            B_peak (float): Peak flux density in Tesla
            frequency (float): Operating frequency in Hz
            mass_core (float): Core mass in kg
            k_h (float): Hysteresis loss coefficient
            k_e (float): Eddy current loss coefficient
            alpha (float): Hysteresis exponent
            beta (float): Eddy current exponent

        Returns:
            float: Total iron loss in Watts
        """
        # Hysteresis loss per unit mass
        P_h_specific = k_h * frequency * (B_peak ** alpha)

        # Eddy current loss per unit mass
        P_e_specific = k_e * (frequency ** 2) * (B_peak ** beta)

        # Total loss
        P_iron = (P_h_specific + P_e_specific) * mass_core

        return P_iron

    @staticmethod
    def compute_copper_loss(I_rms, R_phase, n_phases=3):
        """
        Compute copper winding losses.

        P_cu = n_phases × I_rms² × R_phase

        Args:
            I_rms (float): RMS phase current in Amperes
            R_phase (float): Phase resistance in Ohms
            n_phases (int): Number of phases (default 3)

        Returns:
            float: Copper loss in Watts
        """
        P_cu = n_phases * (I_rms ** 2) * R_phase
        return P_cu


class DemagnetizationChecker:
    """
    Temperature-dependent demagnetization risk assessment.

    Checks if operating flux density exceeds demagnetization threshold
    accounting for temperature effects.
    """

    @staticmethod
    def check_demag_risk(B_operating, B_demag_threshold, temp_C,
                        temp_coeff=-0.12, reference_temp=20.0):
        """
        Check demagnetization risk.

        Temperature-dependent threshold:
        B_demag(T) = B_demag_ref × (1 + temp_coeff × (T - T_ref) / 100)

        Args:
            B_operating (float): Operating reverse flux density in Tesla (positive)
            B_demag_threshold (float): Demagnetization threshold at reference temp
            temp_C (float): Operating temperature in °C
            temp_coeff (float): Temperature coefficient (%/°C)
            reference_temp (float): Reference temperature for threshold (°C)

        Returns:
            tuple: (is_safe: bool, margin_T: float)
                   is_safe: True if B_operating < B_demag(T)
                   margin_T: Margin in Tesla (positive = safe)
        """
        # Adjust threshold for temperature
        delta_temp = temp_C - reference_temp
        temp_factor = 1.0 + temp_coeff * delta_temp / 100.0
        B_demag_adjusted = B_demag_threshold * temp_factor

        # Safety margin
        margin = B_demag_adjusted - B_operating
        is_safe = margin > 0.0

        return is_safe, margin


class EMPINNLoss:
    """
    Differentiable electromagnetic physics constraints for PINN training.

    All methods return torch tensors for gradient-based optimization.
    """

    @staticmethod
    def faraday_residual(flux_density, back_emf, rpm,
                        n_poles=EMConstants.N_POLES,
                        stack_length=EMConstants.STACK_LENGTH):
        """
        Faraday's law residual: ∇×E = -∂B/∂t

        For rotating machine: induced EMF ∝ dΦ/dt = ω × Φ

        Args:
            flux_density (torch.Tensor): Air gap flux density [batch_size]
            back_emf (torch.Tensor): Predicted back-EMF [batch_size]
            rpm (torch.Tensor): Mechanical speed [batch_size]
            n_poles (int): Number of pole pairs
            stack_length (float): Stack length in m

        Returns:
            torch.Tensor: Residual (should be near zero)
        """
        # Expected back-EMF from flux and speed
        # E ∝ n_poles × Φ × ω
        rotor_radius = EMConstants.ROTOR_RADIUS
        pole_arc = EMConstants.POLE_ARC_RATIO * (2.0 * np.pi / n_poles) * rotor_radius
        A_pole = pole_arc * stack_length

        flux = flux_density * A_pole  # Weber
        omega = 2.0 * np.pi * rpm / 60.0  # rad/s

        k_w = 0.96
        E_expected = 1.5 * n_poles * flux * k_w * omega / np.pi

        residual = torch.abs(back_emf - E_expected)
        return residual

    @staticmethod
    def ampere_residual(torque, current, flux_density,
                       n_poles=EMConstants.N_POLES,
                       stack_length=EMConstants.STACK_LENGTH):
        """
        Ampere's law residual: F = J × B (Lorentz force consistency)

        Torque from current and flux: T = (3/2) × n_poles × Φ × I

        Args:
            torque (torch.Tensor): Predicted torque [batch_size]
            current (torch.Tensor): Phase current [batch_size]
            flux_density (torch.Tensor): Air gap flux density [batch_size]
            n_poles (int): Number of pole pairs
            stack_length (float): Stack length in m

        Returns:
            torch.Tensor: Residual (should be near zero)
        """
        rotor_radius = EMConstants.ROTOR_RADIUS
        pole_arc = EMConstants.POLE_ARC_RATIO * (2.0 * np.pi / n_poles) * rotor_radius
        A_pole = pole_arc * stack_length

        flux = flux_density * A_pole
        k_w = 0.96
        k_t = 1.5 * n_poles * flux * k_w

        T_expected = k_t * current

        residual = torch.abs(torque - T_expected)
        return residual

    @staticmethod
    def flux_conservation(flux_in, flux_out):
        """
        Flux conservation constraint: ∇·B = 0

        Flux entering a closed surface equals flux exiting.

        Args:
            flux_in (torch.Tensor): Flux entering [batch_size]
            flux_out (torch.Tensor): Flux exiting [batch_size]

        Returns:
            torch.Tensor: Residual (should be near zero)
        """
        residual = torch.abs(flux_in - flux_out)
        return residual

    @staticmethod
    def saturation_penalty(B_pred, B_sat=1.8):
        """
        Soft penalty for magnetic saturation.

        Penalizes B > B_sat to encourage physical predictions.

        Args:
            B_pred (torch.Tensor): Predicted flux density [batch_size]
            B_sat (float): Saturation threshold in Tesla

        Returns:
            torch.Tensor: Penalty (non-negative, zero if B < B_sat)
        """
        excess = torch.relu(B_pred - B_sat)
        penalty = excess ** 2
        return penalty

    @staticmethod
    def demag_penalty(B_pred, B_threshold):
        """
        Soft penalty for demagnetization risk.

        Penalizes B < -B_threshold (reverse field) to prevent demag.

        Args:
            B_pred (torch.Tensor): Predicted flux density [batch_size]
            B_threshold (float): Demagnetization threshold (positive value)

        Returns:
            torch.Tensor: Penalty (non-negative, zero if B > -B_threshold)
        """
        reverse_excess = torch.relu(-B_pred - B_threshold)
        penalty = reverse_excess ** 2
        return penalty

    @staticmethod
    def cogging_limit(cogging_pred, rated_torque, max_pct=2.0):
        """
        Penalty for excessive cogging torque.

        T_cog should be < max_pct% of rated torque.

        Args:
            cogging_pred (torch.Tensor): Predicted cogging torque [batch_size]
            rated_torque (float): Rated average torque in N⋅m
            max_pct (float): Maximum allowable cogging as % of rated (default 2%)

        Returns:
            torch.Tensor: Penalty (non-negative)
        """
        max_cogging = (max_pct / 100.0) * rated_torque
        excess = torch.relu(torch.abs(cogging_pred) - max_cogging)
        penalty = excess ** 2
        return penalty

    @staticmethod
    def thd_limit(thd_pred, max_thd=3.0):
        """
        Penalty for excessive back-EMF THD.

        Args:
            thd_pred (torch.Tensor): Predicted THD [batch_size]
            max_thd (float): Maximum allowable THD in percent (default 3%)

        Returns:
            torch.Tensor: Penalty (non-negative, zero if THD < max_thd)
        """
        excess = torch.relu(thd_pred - max_thd)
        penalty = excess ** 2
        return penalty

    @staticmethod
    def efficiency_target(eff_pred, target=93.0):
        """
        Penalty for deviation from efficiency target.

        Penalizes both under-performance and over-optimistic predictions.

        Args:
            eff_pred (torch.Tensor): Predicted efficiency [batch_size]
            target (float): Target efficiency in percent (default 93%)

        Returns:
            torch.Tensor: Penalty (non-negative, zero at target)
        """
        deviation = torch.abs(eff_pred - target)
        penalty = deviation ** 2
        return penalty

    @staticmethod
    def combined_em_loss(flux_density, back_emf, torque, current, rpm,
                        cogging_torque, thd, efficiency,
                        rated_torque=15000.0 / (2.0 * np.pi * 150.0 / 60.0),
                        weights=None):
        """
        Combined physics loss for PINN training.

        Weighted sum of all EM constraints and penalties.

        Args:
            flux_density (torch.Tensor): Air gap flux density
            back_emf (torch.Tensor): Back-EMF prediction
            torque (torch.Tensor): Torque prediction
            current (torch.Tensor): Phase current
            rpm (torch.Tensor): Speed
            cogging_torque (torch.Tensor): Cogging torque prediction
            thd (torch.Tensor): THD prediction
            efficiency (torch.Tensor): Efficiency prediction
            rated_torque (float): Rated torque in N⋅m
            weights (dict): Loss weights (defaults to equal weighting)

        Returns:
            torch.Tensor: Scalar combined loss
        """
        if weights is None:
            weights = {
                'faraday': 1.0,
                'ampere': 1.0,
                'saturation': 0.5,
                'demag': 1.0,
                'cogging': 0.3,
                'thd': 0.3,
                'efficiency': 0.5
            }

        # Physics residuals
        L_faraday = EMPINNLoss.faraday_residual(flux_density, back_emf, rpm).mean()
        L_ampere = EMPINNLoss.ampere_residual(torque, current, flux_density).mean()

        # Constraints and penalties
        L_sat = EMPINNLoss.saturation_penalty(flux_density).mean()
        L_demag = EMPINNLoss.demag_penalty(flux_density, EMConstants.DEMAG_THRESHOLD_N48H).mean()
        L_cog = EMPINNLoss.cogging_limit(cogging_torque, rated_torque).mean()
        L_thd = EMPINNLoss.thd_limit(thd).mean()
        L_eff = EMPINNLoss.efficiency_target(efficiency).mean()

        # Combined loss
        total_loss = (weights['faraday'] * L_faraday +
                     weights['ampere'] * L_ampere +
                     weights['saturation'] * L_sat +
                     weights['demag'] * L_demag +
                     weights['cogging'] * L_cog +
                     weights['thd'] * L_thd +
                     weights['efficiency'] * L_eff)

        return total_loss
