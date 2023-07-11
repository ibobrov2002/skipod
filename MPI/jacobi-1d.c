/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <mpi.h>

double bench_t_start, bench_t_end;
int size, rank;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n,
   float A[ n],
   float B[ n])
{
  int i;
  for (i = 0; i < n; i++)
      {
        A[i] = ((float) i+ 2) / n;
        B[i] = ((float) i+ 3) / n;
      }
}

static
void print_array(int n,
   float A[ n])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", A[i]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_jacobi_1d(int tsteps,
       int n,
       float A[ n],
       float B[ n])
{
  int t, i, right_rank = rank + 1, left_rank = rank - 1, count, ibeg, iend;

  MPI_Status status;

  count = n / size;

  ibeg = rank * count + 1;
  if (rank != size - 1) {
    iend = (rank + 1) * count;
  }else{
    iend = n - 1;
  }
  
  if (rank < size) {
    for (t = 0; t < tsteps; t++){
      /*Массив В*/
      for (i = ibeg; i < iend; i++)
        B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
      if (rank != size - 1)
        B[iend] = 0.33333 * (A[iend - 1] + A[iend] + A[iend + 1]);
      if (rank == 0) {
        if (size != 1) {
          MPI_Send(&B[iend], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
          MPI_Recv(&B[iend + 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &status);
        }
      }else if (rank == size - 1) {
        MPI_Send(&B[ibeg], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&B[ibeg - 1], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &status);
      }else {
        MPI_Send(&B[iend], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
        MPI_Send(&B[ibeg], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&B[iend + 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[ibeg - 1], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &status);
      }

      /* Массив A*/
      for (i = ibeg; i < iend; i++)
        A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
      if (rank != size - 1)
        A[iend] = 0.33333 * (B[iend-1] + B[iend] + B[iend + 1]);
      if (rank == 0) {
        if (size != 1) {
          MPI_Send(&A[iend], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
          MPI_Recv(&A[iend + 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &status);
        }
      }else if (rank == size - 1) {
        MPI_Send(&A[ibeg], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&A[ibeg - 1], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &status);
      }else {
        MPI_Send(&A[iend], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
        MPI_Send(&A[ibeg], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&A[iend + 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&A[ibeg - 1], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &status);
      }
    }
  }

  if (size != 1 && rank < size) {
    if (rank != size - 1) {
      MPI_Send(&B[ibeg], iend - ibeg + 1, MPI_DOUBLE, size - 1 , 0, MPI_COMM_WORLD);
    }else {
      int i;
      for (i = 0; i < size - 1; i++) {
        MPI_Recv(&B[i * count + 1], count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);    
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  /*Собираем новые данные на нити size - 1*/
  if (size != 1 && rank < size) {
    if (rank != size - 1) {
      MPI_Send(&A[ibeg], iend - ibeg + 1, MPI_DOUBLE, size - 1 , 0, MPI_COMM_WORLD);
    }else {
      int i;
      for (i = 0; i < size - 1; i++) {
        MPI_Recv(&A[i * count + 1], count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);    
      }
    }
  }  
}


int main(int argc, char** argv)
{
  int n = N;
  int tsteps = TSTEPS;
  float (*A)[n]; 
  float (*B)[n]; 

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > n){
    size = n-3;
  }

  if (rank == size - 1) {
    printf("n=%d tsteps=%d threads=%d\n", n, tsteps, size);
  }

  A = (float(*)[n])malloc ((n) * sizeof(float));
  B = (float(*)[n])malloc ((n) * sizeof(float));

  init_array (n, *A, *B);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == size - 1) {
    bench_timer_start();
  }

  kernel_jacobi_1d(tsteps, n, *A, *B);

  MPI_Barrier(MPI_COMM_WORLD);
  
  if (rank == size - 1) {
    bench_timer_stop();
    bench_timer_print();

    /*print_array(n, *A);
    print_array(n, *B);*/
  }

  free((void*)A);;
  free((void*)B);;

  MPI_Finalize();
  return 0;
}
