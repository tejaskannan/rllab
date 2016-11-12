from rllab.algos.npo import NPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)



class TRPO_Shuffling(TRPO):

    def __init__(self, probability=0, optimizer=None, optimizer_args=None, **kwargs):
        self.probability = probability
        super(optimizer=optimizer, optimizer_args=optimizer_args, **kwargs)

    def optimize(self, paths, samples_data, logger):
        self.log_diagnostics(paths)
        self.optimize_policy(itr, samples_data)
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr, samples_data)
        params["algo"] = self
        if self.store_paths:
            params["paths"] = samples_data["paths"]
        logger.save_itr_params(itr, params)
        logger.log("saved")
        logger.dump_tabular(with_prefix=False)
        if self.plot:
            self.update_plot()
            if self.pause_for_plot:
                input("Plotting evaluation run: Press Enter to "
                          "continue...")


    def train(self):
        self.start_worker()
        self.init_opt()

        itr = self.current_itr
        while itr < self.n_iter:
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples(itr)
                # check data type of paths -> might need to shuffle the list of lists
                samples_data = self.sampler.process_samples(itr, paths)

                # shuffle with given probability
                r = random.random()
                if r < self.probability and itr < self.n_itr - 1:
                    # logger update file
                    self.current_itr = itr + 2
                    next_paths = self.sampler.obtain_samples(itr+1)
                    next_samples_data = self.sampler.process_samples(itr+1, next_paths)
                    optimize(next_paths, next_samples_data, logger)
                    itr = itr + 1
                else:
                    self.current_itr = itr + 1

                optimize(paths, samples_data, logger)
                itr = itr + 1
                
        self.shutdown_worker()