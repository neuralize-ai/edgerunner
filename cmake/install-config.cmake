include(CMakeFindDependencyMacro)
find_dependency(fmt)
find_dependency(span-lite)

include("${CMAKE_CURRENT_LIST_DIR}/edgerunnerTargets.cmake")
