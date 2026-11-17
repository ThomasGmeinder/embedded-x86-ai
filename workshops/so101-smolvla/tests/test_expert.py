from so101_smolvla.expert import Expert
from so101_smolvla.sim import HOLD_STEPS, ArmSim


def test_expert_places_cube_and_holds():
    sim = ArmSim()
    try:
        expert = Expert(sim)
        assert expert.run((0.22, -0.08), azimuth=0.4) is True
        assert sim.in_box()
        assert HOLD_STEPS == int(round(1.5 * 480))
    finally:
        sim.close()
