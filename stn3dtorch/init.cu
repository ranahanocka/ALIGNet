#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "BilinearSamplerVolumetric.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcuvolstn(lua_State *L);

int luaopen_libcuvolstn(lua_State *L)
{
  lua_newtable(L);
  cunn_BilinearSamplerVolumetric_init(L);
  return 1;
}

