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

    int myrank, position;
    int with_leader;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // get rank of the process

    int side = (int)sqrt(N); // square array -> side length is square root of N (N is a square number)

    int recvsize;
    recvsize = 2 * side; // as in 9 stencil, 2 rows/columns are communicated
    
    double **data_new = (double **)malloc(side * sizeof(double *));  // new matrix, updated after every timestep
    
    for (int i = 0; i < side; i++)
    {
        data_new[i] = (double *)malloc(side * sizeof(double));
       
    }
    
    double **data_old = (double **)malloc(side * sizeof(double *));    // data_old stores the data at the start of the time step
    for(int i = 0; i < side; i++)
    {
        data_old[i] = (double *)malloc(side * sizeof(double)); 
    }
    
    for(int i =0 ; i < side; i ++) for(int j = 0; j < side; j ++) data_old[i][j] = 100000;

    // double **data = (double **)malloc(side * sizeof(double *));    // useful for saving data for the with leader case
    // for(int i = 0; i < side; i++)
    // {
    //     data[i] = (double *)malloc(side * sizeof(double));
    // } 
    MPI_Status status;

    // send/recv buffers for MPI, size is 2*side for stencil == 9 case
    double *sendbuffer = (double *)malloc(2 * side * sizeof(double));
    double *recvbuf_right = (double *)malloc(2 * side * sizeof(double));
    double *recvbuf_left = (double *)malloc(2 * side * sizeof(double));
    double *recvbuf_below = (double *)malloc(2 * side * sizeof(double));
    double *recvbuf_above = (double *)malloc(2 * side * sizeof(double));
    int color; // for splitting

    // random initialization in first step
    srand(seed * (myrank + 10));
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < side; j++)
        {

            data_old[i][j] = abs(rand() + (i * rand() + j * myrank));
            // data_old[i][j] = 1;
            data_new[i][j] = 0;
            // data[i][j] = data_old[i][j];
        }
    }

    int **count = (int **)malloc(side * sizeof(int *));
    for (int i = 0; i < side; i++)
    {
        count[i] = (int *)malloc(side * sizeof(int));
    }
    // count stores the number of data points added to get the value
    // in data_new[][], used because edge and corner points would not
    // have 9 values added together e.g. in stencil == 9 case
    // corner would only have 5 values added
    color = myrank / Px;  // will be used in split
    int comm_rank;
    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &newcomm);  // new communicators, for processes in a row in virtual topology 
    stime = MPI_Wtime();
    // for each time step
    MPI_Comm_rank(newcomm, &comm_rank);
    double leader_send_buffer[Px * side * 2];
    double leader_recv_buffer[Px * side * 2];
   
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


        // send data (last column/last two columns) to the process to the right
        if (myrank % Px < Px - 1)
        { // i.e. myrank%Px != Px - 1 i.e. this is not the last process in the "row"
            // last process in the "row" won't have anyone to send it to
            // afterwards, pack the column data points
            // for stencil == 9, we pack the last column first, and then the second last column

            position = 0;
            for (int i = 0; i < side; i++)
            {
                MPI_Pack(&data_old[i][side - 1], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            }
            for (int i = 0; i < side; i++) // pack the second last column
                MPI_Pack(&data_old[i][side - 2], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            // send to process to the right, then receive from the same process
            // for the sending of the 1st column, we receive first and then send

            MPI_Send(sendbuffer, position, MPI_PACKED, myrank + 1, myrank, MPI_COMM_WORLD);

            MPI_Recv(recvbuf_right, recvsize, MPI_DOUBLE, myrank + 1, myrank + 1, MPI_COMM_WORLD, &status);

            for (int i = 0; i < side; i++)
            {
                data_new[i][side - 1] += recvbuf_right[i]; // add 1st col of right to last col of this
                count[i][side - 1]++;
            }

            for (int i = 0; i < side; i++)
            {
                data_new[i][side - 1] += recvbuf_right[i + side]; // add 2nd col of right to last col of this
                count[i][side - 1]++;
                data_new[i][side - 2] += recvbuf_right[i]; // add 1st col of right to 2nd last col of this
                count[i][side - 2]++;
            }
        }
        if (myrank % Px > 0)
        {
            // i.e. myrank%Px != 0 i.e. this is not the first process in the "row"
            // same as before, pack data from columns

            position = 0;
            for (int i = 0; i < side; i++) // pack the first column
            {
                MPI_Pack(&data_old[i][0], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            }
            for (int i = 0; i < side; i++) // pack the second column
            {
                MPI_Pack(&data_old[i][1], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
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
            for (int i = 0; i < side; i++)
            {
                data_new[i][0] += recvbuf_left[i + side];
                count[i][0]++;
                data_new[i][1] += recvbuf_left[i];
                count[i][1]++;
            }
        }

        // communication between same row processes is done, now we do the same for the columns
        // send 2 rows to the process below in the "column"
        if ((myrank / Px) < Py - 1)
        {
            // send first, then  receive
            //we need to pack data because rows are not contiguous in memory
            // so we pack them into a buffer and then do a gather so that the leader has the data
            int position = 0;
            MPI_Pack(&data_old[side - 2][0],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );
            MPI_Pack(&data_old[side - 1][0],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );
            //do gather at the leader
            MPI_Gather(sendbuffer, side * 2, MPI_DOUBLE, leader_send_buffer, side * 2, MPI_DOUBLE, 0, newcomm);

            if (comm_rank == 0)
            {
                MPI_Send(leader_send_buffer, Px * side * 2, MPI_DOUBLE, myrank + Px, myrank, MPI_COMM_WORLD);

                MPI_Recv(leader_recv_buffer, Px * side * 2, MPI_DOUBLE, myrank + Px,myrank + Px ,MPI_COMM_WORLD, &status);
            }
            MPI_Scatter(leader_recv_buffer, side * 2, MPI_DOUBLE, recvbuf_below, side * 2, MPI_DOUBLE, 0, newcomm);
            //now we need to scatter the data recd from the leader below to the respective processes
            for (int j = 0; j < side; j++) 
            {
                data_new[side - 1][j] += recvbuf_below[j];// add first row of below to last 2 rows
                data_new[side - 1][j] += recvbuf_below[j + side];
                data_new[side - 2][j] += recvbuf_below[j];
                count[side - 1][j]++;
                count[side - 2][j]++;
                count[side - 1][j]++;
            }
        }
        // send first row()

        if ((myrank / Px) > 0)
        {
            //leader recv data from the process above
            if(comm_rank == 0) MPI_Recv(leader_recv_buffer, Px * side * 2, MPI_DOUBLE, myrank - Px, myrank - Px, MPI_COMM_WORLD, &status);
            
            MPI_Scatter(leader_recv_buffer, side * 2, MPI_DOUBLE, recvbuf_above, side * 2, MPI_DOUBLE, 0, newcomm);
            //packing first 2 rows of data_old into a buffer
            int position = 0;  // this packing is necessary since on malloc, rows of data may not be contiguous in memory 
            MPI_Pack(data_old[0],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );
            MPI_Pack(data_old[1],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );
            //do gather at the leader
            MPI_Gather(sendbuffer, side * 2, MPI_DOUBLE, leader_send_buffer, side * 2, MPI_DOUBLE, 0, newcomm);
           if(comm_rank==0) MPI_Send(leader_send_buffer, Px * side * 2, MPI_DOUBLE, myrank - Px, myrank, MPI_COMM_WORLD);
            //sending for leader to the process above
            for (int j = 0; j < side; j++)
            {
                data_new[0][j] += recvbuf_above[j];        // when stencil = 9, is *second last* row of top
                data_new[0][j] += recvbuf_above[j + side]; // when stencil = 9, is *last* row of top
                data_new[1][j] += recvbuf_above[j + side]; // when stencil = 9, is *last* row of top
                count[0][j]++;
                count[1][j]++;
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
    // for(int i = 0; i < side; i++)
    // {
    //     for(int j = 0; j < side; j++)
    //     {
    //         printf("%f ", data_new[i][j]);
    //     }
    //     printf("\n");
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    double max_time, time;
    time = etime - stime; // time taken by each process
    // get max time from all processes, process 0 will be the root
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myrank == 0){
        printf("Time with leader = %f\n", max_time);
        // printf("Data=%f\n",data_new[0][0]);
    }

    // for without leader case
    //srand with same seed as before will genrate the same sequence of numbers
    srand(seed * (myrank + 10));
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < side; j++)
        {
            data_old[i][j] = abs(rand() + (i * rand() + j * myrank));  // restoring data_old so as to test for without leader case
           
            data_new[i][j] = 0;
        }
    }

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

        // send data (last column/last two columns) to the process to the right
        if (myrank % Px < Px - 1)
        { // i.e. myrank%Px != Px - 1 i.e. this is not the last process in the "row"
            // last process in the "row" won't have anyone to send it to
            // afterwards, pack the column data points
            // for stencil == 9, we pack the last column first, and then the second last column

            position = 0;
            for (int i = 0; i < side; i++)
            {
                MPI_Pack(&data_old[i][side - 1], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            }
            for (int i = 0; i < side; i++) // pack the second last column
                MPI_Pack(&data_old[i][side - 2], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            // }
            // send to process to the right, then receive from the same process
            // for the sending of the 1st column, we receive first and then send

            MPI_Send(sendbuffer, position, MPI_PACKED, myrank + 1, myrank, MPI_COMM_WORLD);

            MPI_Recv(recvbuf_right, recvsize, MPI_DOUBLE, myrank + 1, myrank + 1, MPI_COMM_WORLD, &status);

            for (int i = 0; i < side; i++)
            {
                data_new[i][side - 1] += recvbuf_right[i]; // add 1st col of right to last col of this
                
                count[i][side - 1]++;
            }

            for (int i = 0; i < side; i++)
            {
                data_new[i][side - 1] += recvbuf_right[i + side]; // add 2nd col of right to last col of this
                
                count[i][side - 1]++;
                data_new[i][side - 2] += recvbuf_right[i]; // add 1st col of right to 2nd last col of this
                
                count[i][side - 2]++;
            }
        }
        if (myrank % Px > 0)
        {
            // i.e. myrank%Px != 0 i.e. this is not the first process in the "row"
            // same as before, pack data from columns

            position = 0;
            for (int i = 0; i < side; i++) // pack the first column
            {
                MPI_Pack(&data_old[i][0], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
            }
            for (int i = 0; i < side; i++) // pack the second column
            {
                MPI_Pack(&data_old[i][1], 1, MPI_DOUBLE, sendbuffer, 2 * side * 8, &position, MPI_COMM_WORLD);
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
            for (int i = 0; i < side; i++)
            {
                data_new[i][0] += recvbuf_left[i + side];
                
                count[i][0]++;
                data_new[i][1] += recvbuf_left[i];
               
                count[i][1]++;
            }
        }

        // communication between same row processes is done, now we do the same for the columns
        // send last 1 or 2 rows to the process below in the "column"

        if ((myrank / Px) < Py - 1)
        {
            // no need to pack here because the data is contiguous
            // send first, then receive
            int position = 0; //we need to pack data because rows need  not be contiguous in memory on malloc
            MPI_Pack(&data_old[side - 2][0],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );
            MPI_Pack(&data_old[side - 1][0],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );

            MPI_Send(sendbuffer, side * 2, MPI_DOUBLE, myrank + Px, myrank, MPI_COMM_WORLD);

            MPI_Recv(recvbuf_below, recvsize, MPI_DOUBLE, myrank + Px, myrank + Px, MPI_COMM_WORLD, &status);

            for (int j = 0; j < side; j++) // add first row of below to last row of this(first 2 rows in case of 9 point)
            {
                data_new[side - 1][j] += recvbuf_below[j];
                data_new[side - 1][j] += recvbuf_below[j + side];
                data_new[side - 2][j] += recvbuf_below[j];
                
                count[side - 1][j]++;
                count[side - 2][j]++;
                count[side - 1][j]++;
            }
        }

        if ((myrank / Px) > 0)
        {

            MPI_Recv(recvbuf_above, recvsize, MPI_DOUBLE, myrank - Px, myrank - Px, MPI_COMM_WORLD, &status);

            int position = 0; //we need to pack data because rows need not be contiguous in memory on malloc
            MPI_Pack(data_old[0],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );
            MPI_Pack(data_old[1],side,MPI_DOUBLE, sendbuffer, 2 * side * sizeof(double), &position, MPI_COMM_WORLD );

            MPI_Send(sendbuffer, side * 2, MPI_DOUBLE, myrank - Px, myrank, MPI_COMM_WORLD);

            for (int j = 0; j < side; j++)
            {
                data_new[0][j] += recvbuf_above[j];        // when stencil = 9, is *second last* row of top
                data_new[0][j] += recvbuf_above[j + side]; //*last* row of top
                data_new[1][j] += recvbuf_above[j + side];
                count[0][j]++;
                count[1][j]++;
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
    // for(int i = 0; i < side; i++)
    // {
    //     for(int j = 0; j < side; j++)
    //     {
    //         printf("%f ", data_new[i][j]);
    //     }
    //     printf("\n");
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    time = etime - stime; // time taken by each process
    // get max time from all processes, process 0 will be the root
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myrank == 0){
        printf("Time without leader = %f\n", max_time);
        printf("Data=%f\n",data_old[0][0]);
    }

    for(int i = 0; i < side; i++)
    {
        free(data_new[i]);
        free(data_old[i]);
        // free(data[i]);
        free(count[i]);
    }
    free(sendbuffer);
    free(recvbuf_right);
    free(recvbuf_left);
    free(recvbuf_below);
    free(recvbuf_above);
    free(count);
    free(data_new);
    free(data_old);
    MPI_Comm_free(&newcomm);
    // free(data);

    MPI_Finalize();
}  