from brian2 import PoissonGroup, Synapses, NeuronGroup, units
import numpy as np


def strong_mem_noise(tr, GExc: NeuronGroup, GInh: NeuronGroup) -> [int]:
    """
        Special kind of Poisson membrane noise that is so strong that it immediately enables
        the post-synaptic neuron.

        Related parameters:

        - :data:`net.standard_params.strong_mem_noise_active`
        - :data:`net.standard_params.strong_mem_noise_rate`
    """

    def strong_mem_noise_on_group(G: NeuronGroup):
        GNoise = PoissonGroup(G.N, rates=tr.strong_mem_noise_rate)
        SynNoise = Synapses(source=GNoise, target=G, on_pre="V_post += 20*mV")
        SynNoise.connect(condition="i==j")
        return [GNoise, SynNoise]

    return strong_mem_noise_on_group(GExc) + strong_mem_noise_on_group(GInh)


def synapse_delays(syn_delay: units.ms, syn_delay_windowsize: units.ms, Syn: Synapses, shape):
    """
        Configure pre-synaptic delays. Similar to Brunel 2000 these are uniformly distributed.
        The delay is the center of the uniform distribution, while the window size defines the
        total width of the distribution.

        Related parameters:

        - :data:`net.standard_params.synEE_delay`
        - :data:`net.standard_params.synEE_delay_windowsize`
        - :data:`net.standard_params.synEI_delay`
        - :data:`net.standard_params.synEI_delay_windowsize`
    """
    # need to create these for all synapses, not only the initially active ones
    delays = np.random.uniform(low=syn_delay - syn_delay_windowsize / 2,
                                  high=syn_delay + syn_delay_windowsize / 2,
                                  size=shape)
    Syn.delay = delays
    return delays
