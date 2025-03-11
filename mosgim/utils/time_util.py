import numpy as np
from datetime import datetime


@np.vectorize
def sec_of_day(time: datetime) -> float:
    """
    Возвращает количество секунд с начала дня для заданного времени.

    :param time: Время для расчета.
    :return: Количество секунд с начала дня.
    """
    day_start = time.replace(hour=0, minute=0, second=0, microsecond=0)
    return (time - day_start).total_seconds()


def sec_of_interval(time: datetime, time0: datetime) -> float:
    """
    Возвращает количество секунд между двумя временными метками.

    :param time: Конечное время.
    :param time0: Начальное время.
    :return: Количество секунд между `time` и `time0`.
    """
    return (time - time0).total_seconds()


sec_of_interval = np.vectorize(sec_of_interval, excluded='time0')
