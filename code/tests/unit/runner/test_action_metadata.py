from ovlab_runner.action_metadata import component_names


def test_action_labels_follow_explicit_indices_and_joint_representation():
    assert component_names(7, {'representation':'joint_position'}) == tuple(f'joint_{index}' for index in range(7))
    assert component_names(3, {'translation_indices':[2, 0], 'gripper_indices':[1]}) == ('ty', 'gripper', 'tx')
    assert component_names(4, {'rotation_representation':'quaternion_wxyz', 'rotation_indices':[0,1,2,3]}) == ('qw','qx','qy','qz')
    assert component_names(2) == ('component_0', 'component_1')
