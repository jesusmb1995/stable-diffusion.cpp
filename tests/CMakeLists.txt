include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_backend_sched
  test_backend_sched.cpp
)
target_link_libraries(test_backend_sched
  PRIVATE
    ggml
    stable-diffusion
    GTest::gtest_main
)
target_include_directories(test_backend_sched
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)
target_compile_features(test_backend_sched PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(test_backend_sched)
