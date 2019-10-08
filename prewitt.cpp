#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <memory.h>

#define MPI_ON //TODO remove these
#ifdef MPI_ON
#include "mpi.h"
#endif

void prewitt(int* image, int* out_image, int rows, int image_width) {
    int process_id=0;
#ifdef MPI_ON
    MPI_Comm_rank( MPI_COMM_WORLD, &process_id);
#endif
    //printf("start prewiit proc:%d rows:%d width:%d\n", process_id, rows, image_width);

    int maskX[3][3];
    maskX[0][0] = +1; maskX[0][1] = 0; maskX[0][2] = -1;
    maskX[1][0] = +1; maskX[1][1] = 0; maskX[1][2] = -1;
    maskX[2][0] = +1; maskX[2][1] = 0; maskX[2][2] = -1;

    /* 3x3 Prewitt mask for Y Dimension. */

    int maskY[3][3];
    maskY[0][0] = +1; maskY[0][1] = +1; maskY[0][2] = +1;
    maskY[1][0] =   0; maskY[1][1] =   0; maskY[1][2] =    0;
    maskY[2][0] =  -1; maskY[2][1] =  -1; maskY[2][2] =  -1;

    for (int row_num = 0; row_num < rows; ++row_num) {
        int grad = 255;
        for (int col_num = 0; col_num < image_width; ++col_num) {
            int grad_x = 0;
            int grad_y = 0;

            /* For handling image boundaries */

            if (row_num == 0 || row_num == (rows - 1) || col_num == 0 || col_num == (image_width - 1))
                //TODO there is a line missing in center of final image
                grad = 0;
            else {
                /* Gradient calculation in rows Dimension */
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        //int ii = inputImage[x + i][y + j];
                        //grad_x += (inputImage[x + i][y + j] * maskX[i + 1][j + 1]);
                        int index = (row_num + i) * image_width + (col_num+j);
                        grad_x += (image[index] * maskX[i + 1][j + 1]);
                    }
                }

                /* Gradient calculation in cols Dimension */
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        //grad_y += (inputImage[x + i][y + j] * maskY[i + 1][j + 1]);
                        int index = (row_num + i) * image_width + (col_num+j);
                        grad_y += (image[index] * maskY[i + 1][j + 1]);
                    }
                }

                /* Gradient magnitude */
                grad = (int) sqrt((grad_x * grad_x) + (grad_y * grad_y));
            }
            int value = grad;
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            int index = (row_num * image_width) + col_num;
            out_image[index] = value;
        }
    }
}

//Process just one row at a time - this is only to patch the final image by the master process
void prewitt_patch(int* image, int* out_image, int rows[], int num_rows, int image_width) {

    int maskX[3][3];
    maskX[0][0] = +1; maskX[0][1] = 0; maskX[0][2] = -1;
    maskX[1][0] = +1; maskX[1][1] = 0; maskX[1][2] = -1;
    maskX[2][0] = +1; maskX[2][1] = 0; maskX[2][2] = -1;

    /* 3x3 Prewitt mask for Y Dimension. */

    int maskY[3][3];
    maskY[0][0] = +1; maskY[0][1] = +1; maskY[0][2] = +1;
    maskY[1][0] =   0; maskY[1][1] =   0; maskY[1][2] =    0;
    maskY[2][0] =  -1; maskY[2][1] =  -1; maskY[2][2] =  -1;

    //for (int row_num = 0; row_num < rows; ++row_num) {
    for (int row_index = 0; row_index < num_rows; ++row_index) {
        int grad = 255;
        for (int col_num = 0; col_num < image_width; ++col_num) {
            int grad_x = 0;
            int grad_y = 0;

            int row_num = rows[row_index];

            /* Gradient calculation in rows Dimension */
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    //int ii = inputImage[x + i][y + j];
                    //grad_x += (inputImage[x + i][y + j] * maskX[i + 1][j + 1]);
                    int index = (row_num + i) * image_width + (col_num+j);
                    grad_x += (image[index] * maskX[i + 1][j + 1]);
                }
            }

            /* Gradient calculation in cols Dimension */
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    //grad_y += (inputImage[x + i][y + j] * maskY[i + 1][j + 1]);
                    int index = (row_num + i) * image_width + (col_num+j);
                    grad_y += (image[index] * maskY[i + 1][j + 1]);
                }
            }

            /* Gradient magnitude */
            grad = (int) sqrt((grad_x * grad_x) + (grad_y * grad_y));

            int value = grad;
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            int index = (row_num * image_width) + col_num;
            out_image[index] = value;
        }
    }

}

void processImage(int* input_image, int* output_image, int image_height, int image_width) {
    //Do NOT use MPI Bcast (Broadcast) to distribute the entire input image to each MPI rank (Hint: Use MPI Scatter).
    //Each MPI rank must only have a chunk of the input image. \
    //Collect processed chunk of the output image using MPI Gather.
#ifdef MPI_ON
    int process_id;
    int numberOfProcesses;
    MPI_Comm_rank( MPI_COMM_WORLD, &process_id);
    MPI_Comm_size( MPI_COMM_WORLD, &numberOfProcesses);
    int image_dim[2];
    image_dim[0] = 0;
    image_dim[1] = 0;
    if (process_id == 0) {
        image_dim[0] = image_height;
        image_dim[1] = image_width;
    }

    //send buffer size to others
    if (process_id == 0) {
        for (int r = 1; r<numberOfProcesses; r++) {
            MPI_Send(&image_dim, 2, MPI_INT, r, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&image_dim, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int num_elements_per_proc = (image_dim[0] / numberOfProcesses);
    int buf_size = image_dim[1] * num_elements_per_proc;

    //printf("===> DIMS id:%d height:%d width:%d elements:%d buf:%d\n", process_id, image_dim[0], image_dim[1], num_elements_per_proc, buf_size);
    int* sub_image_in = (int*) malloc(sizeof(int) * buf_size);
    int* sub_image_out = (int*) malloc(sizeof(int) * buf_size);

    // scatter image

    MPI_Scatter(input_image, buf_size, MPI_INT,
                sub_image_in, buf_size, MPI_INT,
                0, MPI_COMM_WORLD);

    prewitt(sub_image_in, sub_image_out, num_elements_per_proc, image_dim[1]);

    MPI_Gather(sub_image_out, buf_size, MPI_INT,
               output_image, buf_size, MPI_INT,
               0, MPI_COMM_WORLD);

    // patch the individual single rows that the sub proccessors could not analyse since they dont have the adjoing lines
    if (process_id == 0) {
        int* rows = (int*) malloc(sizeof(int) * numberOfProcesses * 2);
        memset(rows, 0, sizeof(int) * numberOfProcesses * 2);
        int j = 0;
        for (int i = 0; i<numberOfProcesses; i++) {
            if (i > 0) {
                rows[j] = num_elements_per_proc * i;
                j++;
            }
            if (i < numberOfProcesses -1) {
                if (j>0) {
                    rows[j] = rows[j - 1] + num_elements_per_proc - 1;
                }
                else {
                    rows[j] = num_elements_per_proc - 1;
                }
                j++;
            }
        }
        prewitt_patch(input_image, output_image, rows, j, image_width);
    }
#else
    prewitt(input_image, output_image, image_height, image_width);
#endif
}

int main_a(int argc, char* argv[]) //TODO
{
    int processId=0, num_processes, image_maxShades, image_height, image_width;
    int *inputImage, *outputImage;

    // Setup MPI
#ifdef MPI_ON
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &processId);
    MPI_Comm_size( MPI_COMM_WORLD, &num_processes);
#endif

    if(argc != 3) {
        if(processId == 0)
            std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename> <Output image filename>" << std::endl;
#ifdef MPI_ON
        MPI_Finalize();
#endif
        return 0;
    }

    if (processId == 0) {
        std::ifstream file(argv[1]);
        if(!file.is_open())
        {
            std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
#ifdef MPI_ON
            MPI_Finalize();
#endif
            return 0;
        }

        std::cout << "Detect edges in " << argv[1] << " using " << num_processes << " processes\n" << std::endl;

        std::string workString;
        /* Remove comments '#' and check image format */
        while(std::getline(file,workString))
        {
            if( workString.at(0) != '#' ){
                if( workString.at(1) != '2' ){
                    std::cout << "Input image is not a valid PGM image" << std::endl;
                    return 0;
                } else {
                    break;
                }
            } else {
                continue;
            }
        }
        /* Check image size */
        while(std::getline(file,workString))
        {
            if( workString.at(0) != '#' ){
                std::stringstream stream(workString);
                int n;
                stream >> n;
                image_width = n;
                stream >> n;
                image_height = n;
                break;
            } else {
                continue;
            }
        }
        inputImage  = new int[image_height*image_width];
        outputImage  = new int[image_height*image_width];

        /* Check image max shades */
        while(std::getline(file,workString))
        {
            if( workString.at(0) != '#' ){
                std::stringstream stream(workString);
                stream >> image_maxShades;
                break;
            } else {
                continue;
            }
        }
        /* Fill input image matrix */
        int pixel_val;
        for( int i = 0; i < image_height; i++ )
        {
            if( std::getline(file,workString) && workString.at(0) != '#' ){
                std::stringstream stream(workString);
                for( int j = 0; j < image_width; j++ ){
                    if( !stream )
                        break;
                    stream >> pixel_val;
                    inputImage[(i*image_width)+j] = pixel_val;
                }
            } else {
                continue;
            }
        }
    } // Done with reading image using process 0

    processImage(inputImage, outputImage, image_height, image_width);

    if (processId == 0) {
        /* Start writing output to your file */
        std::ofstream ofile(argv[2]);
        if( ofile.is_open() )
        {
            ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
            for( int i = 0; i < image_height; i++ )
            {
                for( int j = 0; j < image_width; j++ ){
                    ofile << outputImage[(i*image_width)+j] << " ";
                    //ofile << outputImage[i][j] << " ";
                }
                ofile << "\n";
            }
        } else {
            std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
            return 0;
        }
    }
#ifdef MPI_ON
    MPI_Finalize();
#endif
    return 0;
}
