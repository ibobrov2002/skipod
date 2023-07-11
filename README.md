Для выполнения практических заданий используются суперкомпьютерные вычислительные ресурсы факультета ВМК - [Bluegene/P и Polus].

Реализуемый алгоритм: [1-D Jacobi stencil computation] (jacobi-1d.c)

В задаче требуется:
1. Реализовать параллельную версию алгоритма с использованием технологий OpenMP и MPI (DVMH, OpenACC).
2. Начальные параметры для задачи подбираются таким образом, чтобы:
   * Задача помещалась в оперативную память одного процессора.
   * Время решения задачи было в примерном диапазоне 5 секунд - 15 минут.
3. Исследовать масштабируемость полученной параллельной программы: построить графики зависимости времени исполнения от числа ядер/процессоров для различного объёма входных данных.
Для каждого набора входных данных найти количество ядер/процессоров, при котором время выполнения задачи перестаёт уменьшаться.
Оптимальным является построение трёхмерного графика: по одной из осей время работы программы, по другой - количество ядер/процессоров и по третьей - объём входных данных.
Каждый прогон программы с новыми параметрами рекомендуется выполнять несколько раз с последующим усреднением результата (для избавления от случайных выбросов).
Для замера времени рекомендуется использовать вызовы функции omp_get_wtime или MPI_Wtime, общее время работы должно определяться временем самого медленного из процессов/нитей.
Количество ядер/процессоров рекомендуется задавать в виде p=2n, n=0, 1, 2, ... , k, где k определяется доступными ресурсами.
4. Определить основные причины недостаточной масштабируемости программы при максимальном числе используемых ядер/процессоров.
5. Сравнить эффективность OpenMP и MPI-версий параллельной программы.
6. Подготовить отчет о выполнении задания, включающий: описание реализованного алгоритма, графики зависимости времени исполнения от числа ядер/процессоров для различного объёма входных данных, текст программы.

[Bluegene/P и Polus]: http://hpc.cs.msu.ru/
[1-D Jacobi stencil computation]: http://dvmh.keldysh.ru/attachments/download/5269/jacobi-1d.tgz
