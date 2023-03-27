from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import random

class CyclicLRGiselt_Denis(_LRScheduler):
    r"""Establece la tasa de aprendizaje de cada grupo de parámetros según
    política de tasa de aprendizaje cíclica aleatoria. La politica intercambia valores en una frecuencia aleatoria entre
    
    un minimo y un maximo. Esta politica de tasa de aprendizaja actualiza el Learning Rate despues de cada Epoch

    Argumentos:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Tasa de aprendizaje inicial que es el
            límite inferior del ciclo para cada grupo de parámetros..
        max_lr (float or list): Límites superiores de la tasa de aprendizaje en el ciclo
            para cada grupo de parámetros. Funcionalmente,
            define la amplitud del ciclo (max_lr - base_lr).
            La lr en cualquier ciclo es la suma de base_lr
    y algún escalado de la amplitud; por lo tanto
            max_lr puede no alcanzarse realmente dependiendo de
            función de escalado.
        step_size_up (int): Número de iteraciones de entrenamiento en la
            mitad creciente de un ciclo. Por defecto: 2000
        step_size_down (int): Número de iteraciones de entrenamiento en la
            mitad decreciente de un ciclo. Si step_size_down es None,
            se establece en step_size_up. Por defecto: None
        scale_mode (str): {'chichipi', 'decreciente'}.
            Define si scale_fn se evalúa en
            número de ciclo o iteraciones de ciclo (entrenamiento
            iteraciones desde el inicio del ciclo).
            Por defecto: 'chichipi'
        last_epoch (int): El índice del último lote. Este parámetro se utiliza al
            reanudar un trabajo de entrenamiento. Dado que `step()` debe invocarse después de cada
            lote en lugar de después de cada época, este número representa el número total de *lotes* calculados, no el número total de épocas.
            total de *lotes* calculados, no el número total de épocas calculadas.
            Cuando last_epoch=-1, el programa se inicia desde el principio.
            Por defecto: -1
    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLRGiselt_Denis(optimizer, base_lr=0.01, max_lr=0.1)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 scale_mode='chipichipi',
                 last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        
        self.direction_up = True
        self.half_cycle_steps = 0

        self.base_lr = base_lr
        self.max_lr = max_lr

        self.step_size_up = float(step_size_up)
        self.step_size_down = float(step_size_down) if step_size_down is not None else step_size_up

        self.scale_mode = scale_mode

        super(CyclicLRGiselt_Denis, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        if self.scale_mode == 'decrecimiento':
            if (self.last_epoch == 0):
                lr = self.max_lr
            else: 
                lr = self.get_last_lr()[0] - self.base_lr
                if (lr<0):
                    lr = self.base_lr

        elif self.scale_mode == 'chipichipi':
            if (self.last_epoch == 0):
                last_lr = self.base_lr
                
            else:
                last_lr = self.get_last_lr()[0]

            if (self.half_cycle_steps == self.step_size_up
                and self.direction_up == True):
                lr = random.uniform(last_lr, self.max_lr)
                self.direction_up = False
                self.half_cycle_steps = 1

            elif (self.half_cycle_steps == self.step_size_down
                and self.direction_up == False):
                lr = random.uniform(self.base_lr, last_lr)
                self.direction_up = True
                self.half_cycle_steps = 1

            elif (self.direction_up == True):
                lr = random.uniform(last_lr, self.max_lr)
                self.half_cycle_steps += 1

            elif(self.direction_up == False):
                lr = random.uniform(self.base_lr, last_lr)
                self.half_cycle_steps += 1

        lrs.append(lr)
        return lrs