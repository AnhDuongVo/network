import brian2


def synapse_resolve_dt_correctly(synapse: brian2.Synapses):
    """
        Synapse would incorrectly resolve `dt` if its `dt` were different from
        the NeuronGroup's `dt`.
        For details see https://github.com/brian-team/brian2/pull/1248

        This can be removed once we use an up-to-date brian implementation
        with the fix of the PR. However, this should stay compatible with updates.
    """

    if '_clock_dt' in synapse.variables:
        synapse.variables._variables['dt'] = synapse.variables['_clock_dt']
