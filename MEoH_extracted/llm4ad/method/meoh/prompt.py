from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class MEoHPrompt:
    @classmethod
    def _append_reflection_context(cls, prompt_content: str, reflection_context: str | None) -> str:
        if not reflection_context:
            return f"""
            {prompt_content}
            Do not give additional explanations.
            I'm going to tip $999K for a better heuristics! Let's think step by step.
            """
        return f"""
{prompt_content}
You can use some hints below:
{reflection_context}
Do not give additional explanations.
I'm going to tip $999K for a better heuristics! Let's think step by step.
"""

    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return """
        You are an expert in the domain of optimization heuristics.Your task is to design heuristics that can effectively solve optimization problems.
        """
    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function, reflection_context: str | None = None):
        # 不给已有的算法，获取一个全新的算法。
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)

    @classmethod
    def get_prompt_e1(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        reflection_context: str | None = None,
    ):
        # 给定已有的两个算法，创建一个全新的算法。
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)

    @classmethod
    def get_prompt_e2(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        reflection_context: str | None = None,
    ):
        # 给定指定数量的算法，提取算法的共同思想。
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)

    @classmethod
    def get_prompt_m1(
        cls,
        task_prompt: str,
        indi: Function,
        template_function: Function,
        reflection_context: str | None = None,
    ):
        # 根据指定的算法，进行算法微调。
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # create prmpt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)

    @classmethod
    def get_prompt_m2(
        cls,
        task_prompt: str,
        indi: Function,
        template_function: Function,
        reflection_context: str | None = None,
    ):
        ## 修改超参数
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prmpt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)

    @classmethod
    def get_prompt_hs(
        cls,
        indi: Function,
    ):
        assert hasattr(indi, 'algorithm')
        return f'''You are an expert in code review. Your task is to select only the 5 most important tunable hyperparameters from the function and convert them into default parameters.
[code]
{str(indi)}

Select only the 5 most important threshold, weight, or hardcoded variables that are most likely to affect performance. Convert only these 5 variables into default parameters and give me a 'parameter_ranges' dictionary representation. Key of dict is parameter name. Value of key is a tuple in Python that MUST include 2 float elements: the begin value and the end value for that parameter.

Requirements:
- Do not change the algorithm logic except turning the chosen 5 variables into tunable default parameters.
- Do not output more than 5 parameters in 'parameter_ranges'.
- Prefer parameters that strongly affect solution quality or runtime.
- Keep integer-like control parameters reasonable if you choose them.

- Output code only and enclose your code with Python code block: ```python ... ```.
- Output 'parameter_ranges' dictionary only and enclose your code with other Python code block: ```python ... ```.
'''

    @classmethod
    def get_prompt_p1(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        reflection_context: str | None = None,
    ):
        # 设计性能不变但运行时间更短的算法
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prompt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that maintains the same performance level but has a shorter running time compared to the given ones.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)

    @classmethod
    def get_prompt_t1(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        reflection_context: str | None = None,
    ):
        # 设计运行时间不变但性能更好的算法
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prompt content
        prompt_content = f'''You are an expert in the domain of optimization heuristics.
{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that maintains the same running time but has better performance compared to the given ones.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
'''
        return cls._append_reflection_context(prompt_content, reflection_context)
