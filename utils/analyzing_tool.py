

def get_success_rate(solidification_data_list):
    """
    Get the success rate of the solidification data list.

    Parameters
    ----------
    solidification_data_list : list
        The list of solidification data.

    """
    fail_count = 0

    for each_data in solidification_data_list:
        if len(each_data) == 1:
            if each_data[0]['phase type'] == '':
                fail_count += 1

    success_rate = 1 - fail_count / len(solidification_data_list)

    return success_rate


def sys_each_path_num(path_type, sys_path_type):
    """
    Get the number of each path type in the system path type and sort.

    Parameters
    ----------
    path_type : list
        The list of path types.
    sys_path_type : list
        The list of system path types.

    Returns
    -------
    list
        The list of the system path types.
    list
        The list of the number of each system path type.

    """
    sys_path_num = [0 for _ in range(len(sys_path_type))]

    for each_path in path_type:
        if each_path == ['']:
            continue
        index = sys_path_type.index(each_path)
        sys_path_num[index] += 1

    # Pair the two lists together
    pairs = list(zip(sys_path_type, sys_path_num))

    # Sort the pairs based on the counts
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Unzip the pairs to get the sorted kinds
    sys_path_type, sys_path_num = zip(*pairs)

    return sys_path_type, sys_path_num


def overview_solidification(solidification_data):
    """
    Get the overview of the solidification data.

    Parameters
    ----------
    solidification_data : list
        The list of solidification data.

    Returns
    -------
    list
        The list of the phases in solid.
    list
        The list of the phase fractions in solid.

    """
    phases_in_solid = []

    for step in solidification_data:
        phases = step['phases']
        for key in phases.keys():
            if key not in phases_in_solid and key != 'LIQUID':
                phases_in_solid.append(key)

    phase_fracs_in_solid = [0 for _ in range(len(phases_in_solid))]

    for step in solidification_data:
        phases = step['phases']
        for key in phases.keys():
            if key != 'LIQUID':
                index = phases_in_solid.index(key)
                phase_fracs_in_solid[index] += phases[key][0]

    return phases_in_solid, phase_fracs_in_solid
