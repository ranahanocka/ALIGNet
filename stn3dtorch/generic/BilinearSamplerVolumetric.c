#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BilinearSamplerVolumetric.c"
#else

#include <stdbool.h>
#include <stdio.h>

// Affine Transformation
static int nn_(BilinearSamplerBHWD_updateOutput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_checkudata(L, 4, torch_Tensor);

  int batchsize = inputImages->size[0];
  int inputImages_depth = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int output_height = output->size[2];
  int output_width = output->size[3];
  int output_depth = output->size[1];
  int inputImages_channels = inputImages->size[4];

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[2];
  int output_strideWidth = output->stride[3];
  int output_strideDepth = output->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];
  int inputImages_strideDepth = inputImages->stride[1];
    
  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];
  int grids_strideDepth = grids->stride[1];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THTensor_(data)(inputImages);
  output_data = THTensor_(data)(output);
  grids_data = THTensor_(data)(grids);

  int b, yOut, xOut, zOut;

  for(b=0; b < batchsize; b++)
  {
    for(zOut = 0; zOut < output_depth; zOut++)
    {    
      for(yOut=0; yOut < output_height; yOut++) 
      {
        for(xOut=0; xOut < output_width; xOut++) 
        {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 1];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 2];
        real zf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth];

        // get the weights for interpolation
        int yInTopLeftFront, xInTopLeftFront, zInTopLeftFront;
        real yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeftFront = floor(xcoord);
        xWeightTopLeftFront = 1 - (xcoord - xInTopLeftFront);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeftFront = floor(ycoord);
        yWeightTopLeftFront = 1 - (ycoord - yInTopLeftFront);
            
        real zcoord = (zf + 1) * (inputImages_depth - 1) / 2;
        zInTopLeftFront = floor(zcoord);
        zWeightTopLeftFront = 1 - (zcoord - zInTopLeftFront);

        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut + output_strideDepth * zOut;
        const int inTopLeftFrontAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeftFront 
          + inputImages_strideWidth * xInTopLeftFront + inputImages_strideDepth * zInTopLeftFront;
        
        const int inTopLeftBackAddress = inTopLeftFrontAddress + inputImages_strideDepth;
            
        const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
        const int inTopRightBackAddress = inTopRightFrontAddress + inputImages_strideDepth;
            
        const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
        const int inBottomLeftBackAddress = inBottomLeftFrontAddress + inputImages_strideDepth;
            
        const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;
        const int inBottomRightBackAddress = inBottomRightFrontAddress + inputImages_strideDepth;
            
        real v=0;
        real inTopLeftFront=0;
        real inTopLeftBack=0;
        real inTopRightFront=0;
        real inTopRightBack=0;
        real inBottomLeftFront=0;
        real inBottomLeftBack=0;
        real inBottomRightFront=0;
        real inBottomRightBack=0;

        // we are careful with the boundaries
        bool topLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool topLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1<= inputImages_depth-1);
            
        bool topRightFrontIsIn = xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth - 1;

        bool topRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
        
        bool bottomLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool bottomLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
            
        bool bottomRightFrontIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
    
        bool bottomRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
           if(topLeftFrontIsIn) inTopLeftFront = inputImages_data[inTopLeftFrontAddress + t];
           if(topLeftBackIsIn) inTopLeftBack = inputImages_data[inTopLeftBackAddress + t];
            
           if(topRightFrontIsIn) inTopRightFront = inputImages_data[inTopRightFrontAddress + t];
           if(topRightBackIsIn) inTopRightBack = inputImages_data[inTopRightBackAddress + t];
            
           if(bottomLeftFrontIsIn) inBottomLeftFront = inputImages_data[inBottomLeftFrontAddress + t];
           if(bottomLeftBackIsIn) inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t];
            
           if(bottomRightFrontIsIn) inBottomRightFront = inputImages_data[inBottomRightFrontAddress + t];
           if(bottomRightBackIsIn) inBottomRightBack = inputImages_data[inBottomRightBackAddress + t];

           v = xWeightTopLeftFront * yWeightTopLeftFront * zWeightTopLeftFront * inTopLeftFront
             + xWeightTopLeftFront * yWeightTopLeftFront * (1-zWeightTopLeftFront) * inTopLeftBack
             + (1 - xWeightTopLeftFront) * yWeightTopLeftFront * zWeightTopLeftFront * inTopRightFront
             + (1 - xWeightTopLeftFront) * yWeightTopLeftFront * (1-zWeightTopLeftFront) * inTopRightBack
             + xWeightTopLeftFront * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * inBottomLeftFront
             + xWeightTopLeftFront * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * inBottomLeftBack
             + (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * inBottomRightFront
             + (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * inBottomRightBack;
           
           output_data[outAddress + t] = v;
        }
            
        }
      }
    }
  }

  return 1;
}

static int nn_(BilinearSamplerBHWD_updateGradInput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInputImages = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *gradGrids = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 6, torch_Tensor);

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int inputImages_depth = inputImages->size[1];
    
  int gradOutput_height = gradOutput->size[2];
  int gradOutput_width = gradOutput->size[3];
  int gradOutput_depth = gradOutput->size[1];
  int inputImages_channels = inputImages->size[4];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideHeight = gradOutput->stride[2];
  int gradOutput_strideWidth = gradOutput->stride[3];
  int gradOutput_strideDepth = gradOutput->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];
  int inputImages_strideDepth = inputImages->stride[1];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[2];
  int gradInputImages_strideWidth = gradInputImages->stride[3];
  int gradInputImages_strideDepth = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];
  int grids_strideDepth = grids->stride[1];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[2];
  int gradGrids_strideWidth = gradGrids->stride[3];
  int gradGrids_strideDepth = gradGrids->stride[1];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THTensor_(data)(inputImages);
  gradOutput_data = THTensor_(data)(gradOutput);
  grids_data = THTensor_(data)(grids);
  gradGrids_data = THTensor_(data)(gradGrids);
  gradInputImages_data = THTensor_(data)(gradInputImages);

    int b, yOut, xOut, zOut;

  for(b=0; b < batchsize; b++)
  {
    for(zOut=0; zOut < gradOutput_depth; zOut++)
    {
      for(yOut=0; yOut < gradOutput_height; yOut++)
      {
        for(xOut=0; xOut < gradOutput_width; xOut++)
        {
          //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 1];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 2];
        real zf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth];
        
        // get the weights for interpolation
        int yInTopLeftFront, xInTopLeftFront, zInTopLeftFront;
        real yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeftFront = floor(xcoord);
        xWeightTopLeftFront = 1 - (xcoord - xInTopLeftFront);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeftFront = floor(ycoord);
        yWeightTopLeftFront = 1 - (ycoord - yInTopLeftFront);
            
        real zcoord = (zf + 1) * (inputImages_depth - 1) / 2;
        zInTopLeftFront = floor(zcoord);
        zWeightTopLeftFront = 1 - (zcoord - zInTopLeftFront);
            
        const int inTopLeftFrontAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeftFront 
          + inputImages_strideWidth * xInTopLeftFront + inputImages_strideDepth * zInTopLeftFront;
  
        const int inTopLeftBackAddress = inTopLeftFrontAddress + inputImages_strideDepth;
            
        const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
        const int inTopRightBackAddress = inTopRightFrontAddress + inputImages_strideDepth;
            
        const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
        const int inBottomLeftBackAddress = inBottomLeftFrontAddress + inputImages_strideDepth;
            
        const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;
        const int inBottomRightBackAddress = inBottomRightFrontAddress + inputImages_strideDepth;

        const int gradInputImagesTopLeftFrontAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeftFront 
          + gradInputImages_strideWidth * xInTopLeftFront + gradInputImages_strideDepth * zInTopLeftFront;
        const int gradInputImagesTopLeftBackAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideDepth;
            
        const int gradInputImagesTopRightFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideWidth;
        const int gradInputImagesTopRightBackAddress = gradInputImagesTopRightFrontAddress + gradInputImages_strideDepth;
        
        const int gradInputImagesBottomLeftFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomLeftBackAddress = gradInputImagesBottomLeftFrontAddress +gradInputImages_strideDepth;
            
        const int gradInputImagesBottomRightFrontAddress = gradInputImagesBottomLeftFrontAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomRightBackAddress = gradInputImagesBottomRightFrontAddress + gradInputImages_strideDepth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut 
          + gradOutput_strideWidth * xOut + gradOutput_strideDepth * zOut;

        real topLeftFrontDotProduct = 0;
        real topLeftBackDotProduct = 0;
        real topRightFrontDotProduct = 0;
        real topRightBackDotProduct = 0;
            
        real bottomLeftFrontDotProduct = 0;
        real bottomLeftBackDotProduct = 0;
        real bottomRightFrontDotProduct = 0;
        real bottomRightBackDotProduct = 0;

        real v=0;
        real inTopLeftFront=0;
        real inTopLeftBack=0;
        real inTopRightFront=0;
        real inTopRightBack=0;

        real inBottomLeftFront=0;
        real inBottomLeftBack=0;
        real inBottomRightFront=0;
        real inBottomRightBack=0;
      
        // we are careful with the boundaries
        bool topLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool topLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1<= inputImages_depth-1);
            
        bool topRightFrontIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool topRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
            
        bool bottomLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool bottomLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
            
        bool bottomRightFrontIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
 
        bool bottomRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
        int t;
        
        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftFrontIsIn)
           {
              real inTopLeftFront = inputImages_data[inTopLeftFrontAddress + t];
              topLeftFrontDotProduct += inTopLeftFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftFrontAddress + t] += 
                xWeightTopLeftFront * yWeightTopLeftFront * zWeightTopLeftFront * gradOutValue;
           }
           if(topLeftBackIsIn)
           {
              real inTopLeftBack = inputImages_data[inTopLeftBackAddress + t];
              topLeftBackDotProduct += inTopLeftBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftBackAddress + t] += 
                xWeightTopLeftFront * yWeightTopLeftFront * (1-zWeightTopLeftFront) * gradOutValue;
           }

           if(topRightFrontIsIn)
           {
              real inTopRightFront = inputImages_data[inTopRightFrontAddress + t];
              topRightFrontDotProduct += inTopRightFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightFrontAddress + t] += 
                (1 - xWeightTopLeftFront) * yWeightTopLeftFront * zWeightTopLeftFront * gradOutValue;
           }
           if(topRightBackIsIn)
           {
              real inTopRightBack = inputImages_data[inTopRightBackAddress + t];
              topRightBackDotProduct += inTopRightBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightBackAddress + t] += 
                (1 - xWeightTopLeftFront) * yWeightTopLeftFront * (1-zWeightTopLeftFront) * gradOutValue;
           }
           
           if(bottomLeftFrontIsIn)
           {
              real inBottomLeftFront = inputImages_data[inBottomLeftFrontAddress + t];
              bottomLeftFrontDotProduct += inBottomLeftFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftFrontAddress + t] += 
                xWeightTopLeftFront * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * gradOutValue;
           }
           if(bottomLeftBackIsIn)
           {
              real inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t];
              bottomLeftBackDotProduct += inBottomLeftBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftBackAddress + t] += 
                xWeightTopLeftFront * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * gradOutValue;
           }
      
            if(bottomRightFrontIsIn)
           {
              real inBottomRightFront = inputImages_data[inBottomRightFrontAddress + t];
              bottomRightFrontDotProduct += inBottomRightFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightFrontAddress + t] += 
                (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * gradOutValue;
           }

           if(bottomRightBackIsIn)
           {
              real inBottomRightBack = inputImages_data[inBottomRightBackAddress + t];
              bottomRightBackDotProduct += inBottomRightBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightBackAddress + t] += 
                (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * gradOutValue;
           }
        }

        yf = topLeftFrontDotProduct * xWeightTopLeftFront * zWeightTopLeftFront * (-1)
           + topLeftBackDotProduct * xWeightTopLeftFront * (1-zWeightTopLeftFront) * (-1)
           + topRightFrontDotProduct * (1-xWeightTopLeftFront) * zWeightTopLeftFront * (-1)
           + topRightBackDotProduct * (1-xWeightTopLeftFront) * (1-zWeightTopLeftFront) *(-1)
           + bottomLeftFrontDotProduct * xWeightTopLeftFront * zWeightTopLeftFront * (1)
           + bottomLeftBackDotProduct * xWeightTopLeftFront * (1-zWeightTopLeftFront) * (1)
           + bottomRightFrontDotProduct * (1-xWeightTopLeftFront) * zWeightTopLeftFront * (1)
            + bottomRightBackDotProduct * (1-xWeightTopLeftFront) * (1-zWeightTopLeftFront) *(1);
            
        xf = topLeftFrontDotProduct * yWeightTopLeftFront * zWeightTopLeftFront *(-1)
           + topLeftBackDotProduct * yWeightTopLeftFront * (1-zWeightTopLeftFront) *(-1)
           + topRightFrontDotProduct * yWeightTopLeftFront * zWeightTopLeftFront * 1
           + topRightBackDotProduct * yWeightTopLeftFront * (1-zWeightTopLeftFront) * 1
           + bottomLeftFrontDotProduct * (1-yWeightTopLeftFront) * zWeightTopLeftFront * (-1)
           + bottomLeftBackDotProduct * (1-yWeightTopLeftFront) * (1-zWeightTopLeftFront) * (-1)
           + bottomRightFrontDotProduct * (1-yWeightTopLeftFront) * zWeightTopLeftFront * (1)
            + bottomRightBackDotProduct * (1-yWeightTopLeftFront) *(1-zWeightTopLeftFront) * (1);
            
        zf = topLeftFrontDotProduct * yWeightTopLeftFront * xWeightTopLeftFront * (-1)
           + topLeftBackDotProduct * yWeightTopLeftFront * xWeightTopLeftFront *(1)
           + topRightFrontDotProduct * yWeightTopLeftFront * (1-xWeightTopLeftFront) *(-1)
           + topRightBackDotProduct * yWeightTopLeftFront * (1-xWeightTopLeftFront) *(1)
           + bottomLeftFrontDotProduct * (1-yWeightTopLeftFront) * xWeightTopLeftFront * (-1)
           + bottomLeftBackDotProduct * (1-yWeightTopLeftFront) * xWeightTopLeftFront * (1)
           + bottomRightFrontDotProduct * (1-yWeightTopLeftFront) * (1-xWeightTopLeftFront) *(-1)
            + bottomRightBackDotProduct * (1-yWeightTopLeftFront) * (1-xWeightTopLeftFront) * 1;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth + 1] = yf * (inputImages_height-1) / 2;
        
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth + 2] = xf * (inputImages_width-1) / 2;
            
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth] = zf * (inputImages_depth-1) / 2;
      }
      }
    }
  }
  return 1;
}

// Volumetric Transformation
static int nn_(BilinearSamplerVolumetric_updateOutput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_checkudata(L, 4, torch_Tensor);
  float focal_length = lua_tonumber(L, 5);

  int batchsize = inputImages->size[0];
  int inputImages_depth = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
    
  int output_dist = output->size[1];
  int output_height = output->size[2];
  int output_width = output->size[3];

  int inputImages_channels = inputImages->size[4];

  int output_strideBatch = output->stride[0];
  int output_strideDist = output->stride[1];
  int output_strideHeight = output->stride[2];
  int output_strideWidth = output->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideDepth = inputImages->stride[1];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_strideDepth = grids->stride[1];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THTensor_(data)(inputImages);
  output_data = THTensor_(data)(output);
  grids_data = THTensor_(data)(grids);

  int b, yOut, xOut, disOut;

  for(b=0; b < batchsize; b++)
  {
    for(disOut=0; disOut < output_dist; disOut++)
    {
      for(yOut=0; yOut < output_height; yOut++)
      {
        for(xOut=0; xOut < output_width; xOut++)
        {
           
        //read the grid
        real zf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real yf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];
        real xf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth + 2];
        real disf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth + 3];
  
        //printf("%.3f %.3f %.3f\n", zf, yf, xf);
       // normalize the coordinates (x^w, y^w, z^w, 1)
        //yf = yf / disf;
        //xf = xf / disf;
        //zf = zf / disf - (focal_length + 0.5);

        // get the weights for interpolation
        int zInFrontTopLeft, yInFrontTopLeft, xInFrontTopLeft;
        real zWeightFrontTopLeft, yWeightFrontTopLeft, xWeightFrontTopLeft;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInFrontTopLeft = floor(xcoord);
        xWeightFrontTopLeft = 1 - (xcoord - xInFrontTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInFrontTopLeft = floor(ycoord);
        yWeightFrontTopLeft = 1 - (ycoord - yInFrontTopLeft);

        real zcoord = (zf + 1) * (inputImages_depth - 1) / 2;
        zInFrontTopLeft = floor(zcoord);
        zWeightFrontTopLeft = 1 - (zcoord - zInFrontTopLeft);
           
        const int outAddress = output_strideBatch * b + output_strideDist * disOut + output_strideHeight * yOut + output_strideWidth * xOut;
            
        const int inFrontTopLeftAddress = inputImages_strideBatch * b + inputImages_strideDepth * zInFrontTopLeft + inputImages_strideHeight * yInFrontTopLeft + inputImages_strideWidth * xInFrontTopLeft;
        const int inFrontTopRightAddress = inFrontTopLeftAddress + inputImages_strideWidth;
        const int inFrontBottomLeftAddress = inFrontTopLeftAddress + inputImages_strideHeight;
        const int inFrontBottomRightAddress = inFrontBottomLeftAddress + inputImages_strideWidth;
            
        const int inBackTopLeftAddress = inFrontTopLeftAddress + inputImages_strideDepth;
        const int inBackTopRightAddress = inBackTopLeftAddress + inputImages_strideWidth;
        const int inBackBottomLeftAddress = inBackTopLeftAddress + inputImages_strideHeight;
        const int inBackBottomRightAddress = inBackBottomLeftAddress + inputImages_strideWidth;

        real v=0;
        real inFrontTopLeft=0;
        real inFrontTopRight=0;
        real inFrontBottomLeft=0;
        real inFrontBottomRight=0;
        real inBackTopLeft=0;
        real inBackTopRight=0;
        real inBackBottomLeft=0;
        real inBackBottomRight=0;

        // we are careful with the boundaries
        bool frontTopLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
        bool frontTopRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
        bool frontBottomLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
        bool frontBottomRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
            
        bool backTopLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft+1>= 0 && zInFrontTopLeft+1<= inputImages_depth-1;
        bool backTopRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft+1 >= 0 && zInFrontTopLeft+1 <= inputImages_depth-1;
        bool backBottomLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft+1 >= 0 && zInFrontTopLeft+1 <= inputImages_depth-1;
        bool backBottomRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft+1 >= 0 && zInFrontTopLeft+1 <= inputImages_depth-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
           if(frontTopLeftIsIn) inFrontTopLeft = inputImages_data[inFrontTopLeftAddress + t];
           if(frontTopRightIsIn) inFrontTopRight = inputImages_data[inFrontTopRightAddress + t];
           if(frontBottomLeftIsIn) inFrontBottomLeft = inputImages_data[inFrontBottomLeftAddress + t];
           if(frontBottomRightIsIn) inFrontBottomRight = inputImages_data[inFrontBottomRightAddress + t];
           
           if(backTopLeftIsIn) inBackTopLeft = inputImages_data[inBackTopLeftAddress + t];
           if(backTopRightIsIn) inBackTopRight = inputImages_data[inBackTopRightAddress + t];
           if(backBottomLeftIsIn) inBackBottomLeft = inputImages_data[inBackBottomLeftAddress + t];
           if(backBottomRightIsIn) inBackBottomRight = inputImages_data[inBackBottomRightAddress + t];
           
           v = xWeightFrontTopLeft * yWeightFrontTopLeft * zWeightFrontTopLeft * inFrontTopLeft
             + (1 - xWeightFrontTopLeft) * yWeightFrontTopLeft * zWeightFrontTopLeft * inFrontTopRight
             + xWeightFrontTopLeft * (1 - yWeightFrontTopLeft) * zWeightFrontTopLeft * inFrontBottomLeft
             + (1 - xWeightFrontTopLeft) * (1 - yWeightFrontTopLeft) * zWeightFrontTopLeft * inFrontBottomRight
             + xWeightFrontTopLeft * yWeightFrontTopLeft * (1 - zWeightFrontTopLeft) * inBackTopLeft
             + (1 - xWeightFrontTopLeft) * yWeightFrontTopLeft * (1-zWeightFrontTopLeft) * inBackTopRight
             + xWeightFrontTopLeft * (1 - yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * inBackBottomLeft
             + (1 - xWeightFrontTopLeft) * (1 - yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * inBackBottomRight;
           
           output_data[outAddress + t] = v;
       }

      }
    }
  }
  }
  return 1;
}

static int nn_(BilinearSamplerVolumetric_updateGradInput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInputImages = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *gradGrids = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 6, torch_Tensor);
  float focal_length = lua_tonumber(L, 7);

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_depth = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
    
  int gradOutput_dist = gradOutput->size[1];
  int gradOutput_height = gradOutput->size[2];
  int gradOutput_width = gradOutput->size[3];
    
  int inputImages_channels = inputImages->size[4];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideDist = gradOutput->stride[1];
  int gradOutput_strideHeight = gradOutput->stride[2];
  int gradOutput_strideWidth = gradOutput->stride[3];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideDepth = inputImages->stride[1];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideDepth = gradInputImages->stride[1];
  int gradInputImages_strideHeight = gradInputImages->stride[2];
  int gradInputImages_strideWidth = gradInputImages->stride[3];

  int grids_strideBatch = grids->stride[0];
  int grids_strideDepth = grids->stride[1];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideDepth = gradGrids->stride[1];
  int gradGrids_strideHeight = gradGrids->stride[2];
  int gradGrids_strideWidth = gradGrids->stride[3];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THTensor_(data)(inputImages);
  gradOutput_data = THTensor_(data)(gradOutput);
  grids_data = THTensor_(data)(grids);
  gradGrids_data = THTensor_(data)(gradGrids);
  gradInputImages_data = THTensor_(data)(gradInputImages);

  int b, yOut, xOut, disOut;

  for(b=0; b < batchsize; b++)
  {
    for(disOut = 0; disOut < gradOutput_dist; disOut++)
    {
      for(yOut=0; yOut < gradOutput_height; yOut++)
      {
        for(xOut=0; xOut < gradOutput_width; xOut++)
        {
        
       //read the grid
        real zf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real yf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];
        real xf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth+ 2];
        real disf = grids_data[b*grids_strideBatch + disOut*grids_strideDepth + yOut*grids_strideHeight + xOut*grids_strideWidth+ 3];
       
        // yf = yf / disf;
        //xf = xf / disf;
        //zf = zf / disf;

        // get the weights for interpolation
        int zInFrontTopLeft, yInFrontTopLeft, xInFrontTopLeft;
        real zWeightFrontTopLeft, yWeightFrontTopLeft, xWeightFrontTopLeft;
            
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInFrontTopLeft = floor(xcoord);
        xWeightFrontTopLeft = 1 - (xcoord - xInFrontTopLeft);
            
        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInFrontTopLeft = floor(ycoord);
        yWeightFrontTopLeft = 1 - (ycoord - yInFrontTopLeft);
            
        real zcoord = (zf + 1) * (inputImages_depth - 1) / 2;
        zInFrontTopLeft = floor(zcoord);
        zWeightFrontTopLeft = 1 - (zcoord - zInFrontTopLeft);
           
        const int inFrontTopLeftAddress = inputImages_strideBatch * b + inputImages_strideDepth * zInFrontTopLeft + inputImages_strideHeight * yInFrontTopLeft + inputImages_strideWidth * xInFrontTopLeft;
        const int inFrontTopRightAddress = inFrontTopLeftAddress + inputImages_strideWidth;
        const int inFrontBottomLeftAddress = inFrontTopLeftAddress + inputImages_strideHeight;
        const int inFrontBottomRightAddress = inFrontBottomLeftAddress + inputImages_strideWidth;
            
        const int inBackTopLeftAddress = inFrontTopLeftAddress + inputImages_strideDepth;
        const int inBackTopRightAddress = inBackTopLeftAddress + inputImages_strideWidth;
        const int inBackBottomLeftAddress = inBackTopLeftAddress + inputImages_strideHeight;
        const int inBackBottomRightAddress = inBackBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesFrontTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideDepth * zInFrontTopLeft + gradInputImages_strideHeight * yInFrontTopLeft + gradInputImages_strideWidth * xInFrontTopLeft;
        const int gradInputImagesFrontTopRightAddress = gradInputImagesFrontTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesFrontBottomLeftAddress = gradInputImagesFrontTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesFrontBottomRightAddress = gradInputImagesFrontBottomLeftAddress + gradInputImages_strideWidth;
            
        const int gradInputImagesBackTopLeftAddress = gradInputImagesFrontTopLeftAddress + gradInputImages_strideDepth;
        const int gradInputImagesBackTopRightAddress = gradInputImagesBackTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBackBottomLeftAddress = gradInputImagesBackTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBackBottomRightAddress = gradInputImagesBackBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideDist * disOut + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        real frontTopLeftDotProduct = 0;
        real frontTopRightDotProduct = 0;
        real frontBottomLeftDotProduct = 0;
        real frontBottomRightDotProduct = 0;
        real backTopLeftDotProduct = 0;
        real backTopRightDotProduct = 0;
        real backBottomLeftDotProduct = 0;
        real backBottomRightDotProduct = 0;

        real v=0;
        real inFrontTopLeft=0;
        real inFrontTopRight=0;
        real inFrontBottomLeft=0;
        real inFrontBottomRight=0;
        real inBackTopLeft=0;
        real inBackTopRight=0;
        real inBackBottomLeft=0;
        real inBackBottomRight=0;

        // we are careful with the boundaries
        bool frontTopLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
        bool frontTopRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
        bool frontBottomLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
        bool frontBottomRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft >= 0 && zInFrontTopLeft <= inputImages_depth-1;
            
        bool backTopLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft+1>= 0 && zInFrontTopLeft+1<= inputImages_depth-1;
        bool backTopRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft >= 0 && yInFrontTopLeft <= inputImages_height-1 && zInFrontTopLeft+1 >= 0 && zInFrontTopLeft+1 <= inputImages_depth-1;
        bool backBottomLeftIsIn = xInFrontTopLeft >= 0 && xInFrontTopLeft <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft+1 >= 0 && zInFrontTopLeft+1 <= inputImages_depth-1;
        bool backBottomRightIsIn = xInFrontTopLeft+1 >= 0 && xInFrontTopLeft+1 <= inputImages_width-1 && yInFrontTopLeft+1 >= 0 && yInFrontTopLeft+1 <= inputImages_height-1 && zInFrontTopLeft+1 >= 0 && zInFrontTopLeft+1 <= inputImages_depth-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(frontTopLeftIsIn)
           {
              real inFrontTopLeft = inputImages_data[inFrontTopLeftAddress + t];
              frontTopLeftDotProduct += inFrontTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesFrontTopLeftAddress + t] += xWeightFrontTopLeft * yWeightFrontTopLeft * zWeightFrontTopLeft * gradOutValue;
           }

           if(frontTopRightIsIn)
           {
              real inFrontTopRight = inputImages_data[inFrontTopRightAddress + t];
              frontTopRightDotProduct += inFrontTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesFrontTopRightAddress + t] += (1 - xWeightFrontTopLeft) * yWeightFrontTopLeft * zWeightFrontTopLeft * gradOutValue;
           }

           if(frontBottomLeftIsIn)
           {
              real inFrontBottomLeft = inputImages_data[inFrontBottomLeftAddress + t];
              frontBottomLeftDotProduct += inFrontBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesFrontBottomLeftAddress + t] += xWeightFrontTopLeft * (1 - yWeightFrontTopLeft) * zWeightFrontTopLeft * gradOutValue;
           }

           if(frontBottomRightIsIn)
           {
              real inFrontBottomRight = inputImages_data[inFrontBottomRightAddress + t];
              frontBottomRightDotProduct += inFrontBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesFrontBottomRightAddress + t] += (1 - xWeightFrontTopLeft) * (1 - yWeightFrontTopLeft) * zWeightFrontTopLeft * gradOutValue;
           }
            
           if(backTopLeftIsIn)
            {
                real inBackTopLeft = inputImages_data[inBackTopLeftAddress + t];
                backTopLeftDotProduct += inBackTopLeft * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesBackTopLeftAddress + t] += xWeightFrontTopLeft * yWeightFrontTopLeft * (1-zWeightFrontTopLeft) * gradOutValue;
            }
            
           if(backTopRightIsIn)
            {
                real inBackTopRight = inputImages_data[inBackTopRightAddress + t];
                backTopRightDotProduct += inBackTopRight * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesBackTopRightAddress + t] += (1 - xWeightFrontTopLeft) * yWeightFrontTopLeft * (1-zWeightFrontTopLeft) * gradOutValue;
            }
            
           if(backBottomLeftIsIn)
            {
                real inBackBottomLeft = inputImages_data[inBackBottomLeftAddress + t];
                backBottomLeftDotProduct += inBackBottomLeft * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesBackBottomLeftAddress + t] += xWeightFrontTopLeft * (1 - yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * gradOutValue;
            }
            
           if(backBottomRightIsIn)
            {
                real inBackBottomRight = inputImages_data[inBackBottomRightAddress + t];
                backBottomRightDotProduct += inBackBottomRight * gradOutValue;
                if(!onlyGrid) gradInputImages_data[gradInputImagesBackBottomRightAddress + t] += (1 - xWeightFrontTopLeft) * (1 - yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * gradOutValue;
            }

        }

            
        real dyf = frontTopLeftDotProduct * xWeightFrontTopLeft * zWeightFrontTopLeft * (-1)
            + backTopLeftDotProduct * xWeightFrontTopLeft * (1-zWeightFrontTopLeft) * (-1)
            + frontTopRightDotProduct * (1-xWeightFrontTopLeft) * zWeightFrontTopLeft * (-1)
            + backTopRightDotProduct * (1-xWeightFrontTopLeft) * (1-zWeightFrontTopLeft) *(-1)
            + frontBottomLeftDotProduct * xWeightFrontTopLeft * zWeightFrontTopLeft * (1)
            + backBottomLeftDotProduct * xWeightFrontTopLeft * (1-zWeightFrontTopLeft) * (1)
            + frontBottomRightDotProduct * (1-xWeightFrontTopLeft) * zWeightFrontTopLeft * (1)
            + backBottomRightDotProduct * (1-xWeightFrontTopLeft) * (1-zWeightFrontTopLeft) *(1);
            
        real dxf = frontTopLeftDotProduct * yWeightFrontTopLeft * zWeightFrontTopLeft *(-1)
            + backTopLeftDotProduct * yWeightFrontTopLeft * (1-zWeightFrontTopLeft) *(-1)
            + frontTopRightDotProduct * yWeightFrontTopLeft * zWeightFrontTopLeft * 1
            + backTopRightDotProduct * yWeightFrontTopLeft * (1-zWeightFrontTopLeft) * 1
            + frontBottomLeftDotProduct * (1-yWeightFrontTopLeft) * zWeightFrontTopLeft * (-1)
            + backBottomLeftDotProduct * (1-yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * (-1)
            + frontBottomRightDotProduct * (1-yWeightFrontTopLeft) * zWeightFrontTopLeft * 1
            + backBottomRightDotProduct * (1-yWeightFrontTopLeft) *(1-zWeightFrontTopLeft) * 1;
            
        real dzf = frontTopLeftDotProduct * yWeightFrontTopLeft * xWeightFrontTopLeft * (-1)
            + backTopLeftDotProduct * yWeightFrontTopLeft * xWeightFrontTopLeft *1
            + frontTopRightDotProduct * yWeightFrontTopLeft * (1-xWeightFrontTopLeft) *(-1)
            + backTopRightDotProduct * yWeightFrontTopLeft * (1-xWeightFrontTopLeft) *1
            + frontBottomLeftDotProduct * (1-yWeightFrontTopLeft) * xWeightFrontTopLeft * (-1)
            + backBottomLeftDotProduct * (1-yWeightFrontTopLeft) * xWeightFrontTopLeft * 1
            + frontBottomRightDotProduct * (1-yWeightFrontTopLeft) * (1-xWeightFrontTopLeft) *(-1)
            + backBottomRightDotProduct * (1-yWeightFrontTopLeft) * (1-xWeightFrontTopLeft) * 1;
    
        gradGrids_data[b*gradGrids_strideBatch + disOut*gradGrids_strideDepth + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = dzf * (inputImages_depth-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + disOut*gradGrids_strideDepth + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = dyf * (inputImages_height-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + disOut*gradGrids_strideDepth + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 2] = dxf * (inputImages_width-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + disOut*gradGrids_strideDepth + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 3] = 0;
        //  -(dyf* (inputImages_height-1) / 2*yf + dxf* (inputImages_width-1) / 2*xf + dzf* (inputImages_depth-1) / 2*(zf+focal_length+0.5))/disf;
      }
    }
  }
  }
  return 1;
}

static const struct luaL_Reg nn_(BilinearSamplerVolumetric__) [] = {
  {"BilinearSamplerVolumetric_updateOutput", nn_(BilinearSamplerVolumetric_updateOutput)},
  {"BilinearSamplerVolumetric_updateGradInput", nn_(BilinearSamplerVolumetric_updateGradInput)},
  {"BilinearSamplerBHWD_updateOutput", nn_(BilinearSamplerBHWD_updateOutput)},
  {"BilinearSamplerBHWD_updateGradInput", nn_(BilinearSamplerBHWD_updateGradInput)},
  {NULL, NULL}
};

static void nn_(BilinearSamplerVolumetric_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(BilinearSamplerVolumetric__), "nn");
  lua_pop(L,1);
}

#endif
