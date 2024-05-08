 #include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"


int main(int argc, char *argv[])
{
    int P, Px, Py, N, num_time_steps, seed, stencil;
    double stime, etime;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &P); // number of processes, given in assignment
    Px = atoi(argv[1]);                // number of processes in one "row"
    Py = P / Px;                       // number of processes in one "column"
    N = atoi(argv[2]);                 // number of data points
    num_time_steps = atoi(argv[3]);    // number of time steps
    seed = atoi(argv[4]);              // seed for random number generator
    stencil = atoi(argv[5]);           // 5 point or 9 point stencil

    int myrank, position;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // get rank of the process

    int side = (int)sqrt(N); // square array -> side length is square root of N (N is a square number)

    int recvsize; 
    if (stencil == 5)
        recvsize = side;
    else if (stencil == 9)
        recvsize = 2 * side;  // as in 9 stencil, 2 rows/columns are communicated

    double data_new[side][side];
    double data_old[side][side]; // data_old stores the data at the start of the time step
    
    MPI_Status status;

    // send/recv buffers for MPI, size is 2*side for stencil == 9 case
    // works for stencil == 5 case too so no problem
    double sendbuffer[2 * side];
    double recvbuf_right[2 * side];   
    double recvbuf_left[2 * side];
    double recvbuf_below[2 * side];
    double recvbuf_above[2 * side];

    // random initialization in first step
    srand(seed * (myrank + 10));
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < side; j++)
        {

            data_old[i][j] = abs(rand() + (i * rand() + j * myrank)) / 100;
            data_new[i][j] = 0;
        }
    }

    int count[side][side];
    // count stores the number of data points added to get the value
    // in data_new[][], used because edge and corner points would not
    // have 5 or 9 values added together e.g. in stencil == 5 case
    // corner would only have 3 values added
    stime = MPI_Wtime();
    // for each time step

    for (int n = 1; n <= num_time_steps; n++)
    {

        // initialize counts to 0
        for (int i = 0; i < side; i++)
            for (int j = 0; j < side; j++)
                count[i][j] = 0;
        // we put the if statement outside the for(side) loops so that they are only
        // run once (i.e. a few times) per time step
        // computing new values from data points available in same process
        for (int i = 0; i < side; i++)
        {
            for (int j = 0; j < side; j++)
            {
                data_new[i][j] = data_old[i][j];
                count[i][j]++;
                if (i > 0)
                {
                    data_new[i][j] += data_old[i - 1][j];
                    count[i][j]++;
                }
                if (i < side - 1)
                {
                    data_new[i][j] += data_old[i + 1][j];
                    count[i][j]++;
                }
                if (j > 0)
                {
                    data_new[i][j] += data_old[i][j - 1];
                    count[i][j]++;
                }
                if (j < side - 1)
                {
                    data_new[i][j] += data_old[i][j + 1];
                    count[i][j]++;
                }
            }
        }
        if (stencil == 9)   // for 9 point stencil only
        {
            for (int i = 0; i < side; i++)
            {
                for (int j = 0; j < side; j++)
                {
                    if (j > 1)
                    {
                        data_new[i][j] += data_old[i][j - 2];
                        count[i][j]++;
                    }
                    if (i > 1)
                    {
                        data_new[i][j] += data_old[i - 2][j];
                        count[i][j]++;
                    }
                    if (i < side - 2)
                    {
                        data_new[i][j] += data_old[i + 2][j];
                        count[i][j]++;
                    }
                    if (j < side - 2)
                    {
                        data_new[i][j] += data_old[i][j + 2];
                        count[i][j]++;
                    }
                }
            }
        }
        // send data (last column/last two columns) to the process to the right
        if (myrank % Py < Py - 1)
        { // i.e. myrank%Py != Py - 1 i.e. this is not the last process in the "row"
            // last process in the "row" won't have anyone to send it to
            // afterwards, pack the column data points
            // for stencil == 9, we pack the last column first, and then the second last column

            position = 0;
            if (stencil == 5)
            {
                for (int i = 0; i < side; i++) // pack the last column
                {
                    MPI_Pack(&data_old[i][side - 1], 1, MPI_DOUBLE, sendbuffer, side * 8, &position, MPI_COMM_WORLD); // check how much size required
                }
            }
            else if (stencil == 9)
            {
                for (int i = 0; i < side; i++)
                {
                    MPI_Pack(&data_old[i][side - 1], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
                }
                for (int i = 0; i < side; i++)  // pack the second last column
                    MPI_Pack(&data_old[i][side - 2], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            }
            // send to process to the right, then receive from the same process
            // for the sending of the 1st column, we receive first and then send

            MPI_Send(sendbuffer, position, MPI_PACKED, myrank + 1, myrank, MPI_COMM_WORLD);

            MPI_Recv(recvbuf_right, recvsize, MPI_DOUBLE, myrank + 1, myrank + 1, MPI_COMM_WORLD, &status);

            for (int i = 0; i < side; i++)
            {
                data_new[i][side - 1] += recvbuf_right[i]; // add 1st col of right to last col of this
                count[i][side - 1]++;
            }
            if (stencil == 9)
            {
                for (int i = 0; i < side; i++)
                {
                    data_new[i][side - 1] += recvbuf_right[i + side]; // add 2nd col of right to last col of this
                    count[i][side - 1]++;
                    data_new[i][side - 2] += recvbuf_right[i]; // add 1st col of right to 2nd last col of this
                    count[i][side - 2]++;
                }
            }
        }
        if (myrank % Py > 0)
        {
            // i.e. myrank%Py != 0 i.e. this is not the first process in the "row"
            // same as before, pack data from columns

            position = 0;
            if (stencil == 5)
            {
                for (int i = 0; i < side; i++)  // pack the first column
                {
                    MPI_Pack(&data_old[i][0], 1, MPI_DOUBLE, sendbuffer, side * 8, &position, MPI_COMM_WORLD); // check how much size required
                }
            }
            else if (stencil == 9)
            {
                for (int i = 0; i < side; i++)  // pack the first column
                {
                    MPI_Pack(&data_old[i][0], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
                }
                for (int i = 0; i < side; i++)   // pack the second column 
                {
                    MPI_Pack(&data_old[i][1], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
                }
            }

            // receive from the process to the left, then send to the same process

            MPI_Recv(recvbuf_left, recvsize, MPI_DOUBLE, myrank - 1, myrank - 1, MPI_COMM_WORLD, &status);
            MPI_Send(sendbuffer, position, MPI_PACKED, myrank - 1, myrank, MPI_COMM_WORLD);

            // 1st row of recv buffer: last col of left
            // 2nd row of recv buffer: 2nd last col of left
            for (int i = 0; i < side; i++)
            {
                data_new[i][0] += recvbuf_left[i];
                count[i][0]++;
            }
            if (stencil == 9)
            {
                for (int i = 0; i < side; i++)
                {
                    data_new[i][0] += recvbuf_left[i + side];
                    count[i][0]++;
                    data_new[i][1] += recvbuf_left[i];
                    count[i][1]++;
                }
            }
        }

        // communication between same row processes is done, now we do the same for the columns
        // send last 1 or 2 rows to the process below in the "column"

        if ((myrank / Py) < Px - 1)
        {
            // no need to pack here because the data is contiguous
            // send first, then receive

            if (stencil == 5)
            {
                MPI_Send(&data_old[side - 1][0], side, MPI_DOUBLE, myrank + Py, myrank, MPI_COMM_WORLD);
            }
            if (stencil == 9)
            {
                MPI_Send(&data_old[side - 2][0], side * 2, MPI_DOUBLE, myrank + Py, myrank, MPI_COMM_WORLD);
            }

            MPI_Recv(recvbuf_below, recvsize, MPI_DOUBLE, myrank + Py, myrank + Py, MPI_COMM_WORLD, &status);

            for (int j = 0; j < side; j++)  // add first row of below to last row of this(first 2 rows in case of 9 point)
            {
                data_new[side - 1][j] += recvbuf_below[j];
                if (stencil == 9)
                {
                    data_new[side - 1][j] += recvbuf_below[j + side];
                    data_new[side - 2][j] += recvbuf_below[j];
                    count[side - 1][j]++;
                    count[side - 2][j]++;
                }
                count[side - 1][j]++;
            }
        }
        // send first row()

        if ((myrank / Py) > 0)
        {

            MPI_Recv(recvbuf_above, recvsize, MPI_DOUBLE, myrank - Py, myrank - Py, MPI_COMM_WORLD, &status);
            if (stencil == 5)
                MPI_Send(data_old, side, MPI_DOUBLE, myrank - Py, myrank, MPI_COMM_WORLD);
            if (stencil == 9)
                MPI_Send(data_old, side * 2, MPI_DOUBLE, myrank - Py, myrank, MPI_COMM_WORLD);

            for (int j = 0; j < side; j++)
            {
                data_new[0][j] += recvbuf_above[j]; // when stencil = 5, is last row of top, when stencil = 9, is *second last* row of top
                if (stencil == 9)
                {
                    data_new[0][j] += recvbuf_above[j + side]; //*last* row of top
                    data_new[1][j] += recvbuf_above[j + side];
                    count[0][j]++;
                    count[1][j]++;
                }
                count[0][j]++;
            }
        }
        
        for (int i = 0; i < side; i++)
        {
            for (int j = 0; j < side; j++)
            {
                // divide by count to get average
                data_new[i][j] = data_new[i][j] / count[i][j];
                data_old[i][j] = data_new[i][j]; // Assign new values to old values
                
            }
        }
    }
    etime = MPI_Wtime(); // end time

    
    double max_time, time;
    time = etime - stime;  // time taken by each process
    // get max time from all processes, process 0 will be the root 
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
    if (myrank == 0)
        printf("%f\n", max_time);
    MPI_Finalize();
}


