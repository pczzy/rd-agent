from rdagent.core.developer import Developer
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.oai.llm_utils import md5_hash


class CachedRunner(Developer[ASpecificExp]):
    def get_cache_extra_key(self) -> str:
        """Extra state the cached result depends on beyond the tasks themselves.

        Backtest results depend on the train/valid/test windows, which are NOT part of
        any task description: without this, changing the date settings silently reuses
        results computed for the old windows. Subclasses override it; the lookup happens
        on `self`, so it still applies where the decorator names CachedRunner.get_cache_key.
        """
        return ""

    def get_cache_key(self, exp: Experiment) -> str:
        all_tasks = []
        for based_exp in exp.based_experiments:
            all_tasks.extend(based_exp.sub_tasks)
        all_tasks.extend(exp.sub_tasks)
        task_info_list = [task.get_task_information() for task in all_tasks]
        task_info_str = "\n".join(task_info_list)
        extra_key = self.get_cache_extra_key()
        if extra_key:
            task_info_str = f"{task_info_str}\n{extra_key}"
        return md5_hash(task_info_str)

    def assign_cached_result(self, exp: Experiment, cached_res: Experiment) -> Experiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1].result = cached_res.based_experiments[-1].result
        exp.result = cached_res.result
        return exp
