"""PCSELSim public API / PCSEL 时域耦合波仿真包的公共接口。

Most users should run a script in ``scripts/``.  Import these objects directly
only when building notebooks or automated scans. / 初学者优先运行 ``scripts``
中的主程序；编写批量扫描或 notebook 时再直接导入这里的对象。
"""

from .config import SimulationConfig, load_config
from .solver import SimulationResult, TimeDomainSolver

__all__ = ["SimulationConfig", "SimulationResult", "TimeDomainSolver", "load_config"]
__version__ = "0.1.0"

