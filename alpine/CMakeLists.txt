# -----------------------------------------------------------------------------
# alpine/CMakeLists.txt
#
# This module provides example physics applications using IPPL.
# -----------------------------------------------------------------------------

add_subdirectory(ExamplesWithoutPicManager)

function(add_alpine_example name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name} PRIVATE IPPL::ippl)

  # --- ADDED GINKGO LINKING FOR STANDARD EXAMPLES ---
  if(IPPL_ENABLE_GINKGO)
    target_link_libraries(${name} PRIVATE Ginkgo::ginkgo)
  endif()
  # --------------------------------------------------

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

if(IPPL_ENABLE_TESTS)
  # cmake-format: off
  # Landau will write a CSV file to the data directory that we will validate later
  make_directory("${PROJECT_BINARY_DIR}/alpine/data")
  # Add the test
  add_ippl_integration_test(LandauDamping 
    ARGS "16" "16" "16" "10000000" "25" "FFT" "0.01" "LeapFrog" "--overallocate" "2.0" "--info" "10"
    LABELS alpine integration
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/alpine/")
  # cmake-format: on

  # --- ADDED GINKGO LINKING FOR THE TEST EXECUTABLE ---
  if(IPPL_ENABLE_GINKGO)
    target_link_libraries(LandauDamping PRIVATE Ginkgo::ginkgo)
  endif()
  # ----------------------------------------------------

else()
  add_alpine_example(LandauDamping)
endif()

add_alpine_example(PenningTrap)
add_alpine_example(BumponTailInstability)

add_subdirectory(validation)
