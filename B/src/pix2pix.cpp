#include "pix2pix.h"
#include "control.h"

#include "util.h"

#include <cstring>

Control Cntl;

void pix2pix_init() 
{
  /*
   * You can do input-independent and input-size-independent jobs here.
   * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
   * Execution time of this function is not measured, so do as much as possible!
   */
	
	Cntl.MPI_Init();

	Cntl.Initialize();

}

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image) 
{
  /*
   * !!!!!!!! Caution !!!!!!!!
   * In MPI program, all buffers and num_image are only given to rank 0 process.
   * You should manually:
   *   1. allocate buffers on others
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */
	Cntl.Alloc(input_buf, weight_buf, num_image);

	Cntl.EncodePhase_v2();

	Cntl.DecodePhase_v2();

	Cntl.PushOutput(output_buf);
	
	Cntl.PrintTimerInfo();
}


