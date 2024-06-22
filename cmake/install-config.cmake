include(CMakeFindDependencyMacro)
find_dependency(fmt)
find_dependency(span-lite)
find_dependency(tensorflowlite)

include("${CMAKE_CURRENT_LIST_DIR}/edgerunnerTargets.cmake")
