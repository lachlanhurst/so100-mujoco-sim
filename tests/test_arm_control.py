from so100_mujoco_sim.arm_control import calculate_transition_joint_positions

def test_calculate_transition_joint_positions():

    start_positions = [0.0, 0.0, 0.0]
    end_positions = [1.0, 2.0, 3.0]

    intermediate_steps = calculate_transition_joint_positions(
        start_positions,
        end_positions,
        0.5
    )

    # check start and end of the generated steps are ok
    assert intermediate_steps[0] == start_positions
    assert intermediate_steps[-1] == end_positions

    last_s = intermediate_steps[0]
    for s in intermediate_steps:
        for i in range(len(s)):
            # now check that the change in the angle isn't above what
            # we specified above
            assert abs(s[i] - last_s[i]) <= 0.5
        last_s = s
