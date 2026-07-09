cmake_minimum_required(VERSION 3.26...3.31)

project(LdsPhaseTestHarness LANGUAGES CXX)

set(LDS_GPU_ARCH "" CACHE STRING "Required AMD GPU architecture for HIP benchmarks, e.g. gfx942")
option(LDS_BUILD_HIP_BENCHMARKS "Build HIP benchmarks when a HIP toolchain is available" ON)
option(LDS_REQUIRE_HIP "Fail configuration when HIP benchmarks are enabled but a HIP compiler is unavailable" OFF)

set(ROCM_PATH "" CACHE PATH "ROCm SDK root")

if(ROCM_PATH AND NOT EXISTS "${ROCM_PATH}")
  message(STATUS "Ignoring missing ROCM_PATH: ${ROCM_PATH}")
  set(ROCM_PATH "" CACHE PATH "ROCm SDK root" FORCE)
endif()

function(lds_is_rocm_root path result)
  if(EXISTS "${path}/include/hip/hip_runtime.h" OR
     EXISTS "${path}/lib/llvm/bin/amdclang++" OR
     EXISTS "${path}/bin/hipcc")
    set("${result}" TRUE PARENT_SCOPE)
  else()
    set("${result}" FALSE PARENT_SCOPE)
  endif()
endfunction()

function(lds_rocm_root_has_hip_language_package path result)
  if(NOT path OR NOT EXISTS "${path}")
    set("${result}" FALSE PARENT_SCOPE)
    return()
  endif()

  set(_hip_lang_configs
    "${path}/lib/cmake/hip-lang/hip-lang-config.cmake"
    "${path}/lib64/cmake/hip-lang/hip-lang-config.cmake"
  )
  file(GLOB _arch_hip_lang_configs CONFIGURE_DEPENDS
    "${path}/lib/*/cmake/hip-lang/hip-lang-config.cmake"
  )
  list(APPEND _hip_lang_configs ${_arch_hip_lang_configs})

  foreach(_hip_lang_config IN LISTS _hip_lang_configs)
    if(EXISTS "${_hip_lang_config}")
      set("${result}" TRUE PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set("${result}" FALSE PARENT_SCOPE)
endfunction()

function(lds_infer_rocm_root_from_compiler compiler result)
  set(_inferred_root "")
  if(compiler AND IS_ABSOLUTE "${compiler}")
    get_filename_component(_compiler_dir "${compiler}" DIRECTORY)
    foreach(_candidate IN ITEMS
        "${_compiler_dir}/.."
        "${_compiler_dir}/../.."
        "${_compiler_dir}/../../..")
      get_filename_component(_candidate_abs "${_candidate}" ABSOLUTE)
      lds_is_rocm_root("${_candidate_abs}" _is_rocm_root)
      if(_is_rocm_root)
        set(_inferred_root "${_candidate_abs}")
        break()
      endif()
    endforeach()
  endif()

  set("${result}" "${_inferred_root}" PARENT_SCOPE)
endfunction()

function(lds_normalize_hip_compiler compiler result)
  set(_normalized_compiler "${compiler}")

  if(compiler AND IS_ABSOLUTE "${compiler}")
    get_filename_component(_compiler_name "${compiler}" NAME)
    if(_compiler_name STREQUAL "amdclang++" OR _compiler_name STREQUAL "clang++")
      lds_infer_rocm_root_from_compiler("${compiler}" _compiler_rocm_root)
      if(_compiler_rocm_root)
        foreach(_candidate IN ITEMS
            "${_compiler_rocm_root}/lib/llvm/bin/${_compiler_name}"
            "${_compiler_rocm_root}/lib/llvm/bin/amdclang++")
          if(EXISTS "${_candidate}")
            set(_normalized_compiler "${_candidate}")
            break()
          endif()
        endforeach()
      endif()
    endif()
  endif()

  set("${result}" "${_normalized_compiler}" PARENT_SCOPE)
endfunction()

function(lds_hip_compiler_runs compiler result reason)
  if(NOT compiler)
    set("${result}" FALSE PARENT_SCOPE)
    set("${reason}" "A HIP compiler was not found." PARENT_SCOPE)
    return()
  endif()

  if(IS_ABSOLUTE "${compiler}" AND NOT EXISTS "${compiler}")
    set("${result}" FALSE PARENT_SCOPE)
    set("${reason}" "CMAKE_HIP_COMPILER does not exist: ${compiler}" PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND "${compiler}" --version
    RESULT_VARIABLE _compiler_result
    OUTPUT_VARIABLE _compiler_stdout
    ERROR_VARIABLE _compiler_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )

  if(_compiler_result EQUAL 0)
    set("${result}" TRUE PARENT_SCOPE)
    set("${reason}" "" PARENT_SCOPE)
    return()
  endif()

  set(_message "HIP compiler ${compiler} failed to run")
  if(NOT "${_compiler_stderr}" STREQUAL "")
    string(REPLACE "\n" " " _compiler_stderr_one_line "${_compiler_stderr}")
    string(APPEND _message ": ${_compiler_stderr_one_line}")
  elseif(NOT "${_compiler_stdout}" STREQUAL "")
    string(REPLACE "\n" " " _compiler_stdout_one_line "${_compiler_stdout}")
    string(APPEND _message ": ${_compiler_stdout_one_line}")
  else()
    string(APPEND _message ".")
  endif()

  set("${result}" FALSE PARENT_SCOPE)
  set("${reason}" "${_message}" PARENT_SCOPE)
endfunction()

set(_LDS_ROCM_ROOT_HINTS)
file(GLOB _LDS_VENV_ROCM_DEVEL_ROOTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python*/site-packages/_rocm_sdk_devel"
)
list(SORT _LDS_VENV_ROCM_DEVEL_ROOTS)
foreach(_root IN LISTS _LDS_VENV_ROCM_DEVEL_ROOTS)
  list(APPEND _LDS_ROCM_ROOT_HINTS "${_root}")
endforeach()

file(GLOB _LDS_VENV_ROCM_CORE_ROOTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python*/site-packages/_rocm_sdk_core"
)
list(SORT _LDS_VENV_ROCM_CORE_ROOTS)
foreach(_root IN LISTS _LDS_VENV_ROCM_CORE_ROOTS)
  list(APPEND _LDS_ROCM_ROOT_HINTS "${_root}")
endforeach()

foreach(_root IN ITEMS "$ENV{ROCM_PATH}" "$ENV{HIP_PATH}" "$ENV{ROCM_SDK_ROOT}")
  if(NOT "${_root}" STREQUAL "")
    list(APPEND _LDS_ROCM_ROOT_HINTS "${_root}")
  endif()
endforeach()

if(EXISTS "/opt/rocm")
  list(APPEND _LDS_ROCM_ROOT_HINTS "/opt/rocm")
endif()

if(_LDS_ROCM_ROOT_HINTS)
  list(REMOVE_DUPLICATES _LDS_ROCM_ROOT_HINTS)
endif()

if(NOT ROCM_PATH)
  foreach(_root IN LISTS _LDS_ROCM_ROOT_HINTS)
    lds_is_rocm_root("${_root}" _is_rocm_root)
    if(_is_rocm_root)
      set(ROCM_PATH "${_root}" CACHE PATH "ROCm SDK root" FORCE)
      break()
    endif()
  endforeach()
endif()

set(_LDS_HIP_COMPILER_HINTS)
if(ROCM_PATH)
  list(APPEND _LDS_HIP_COMPILER_HINTS "${ROCM_PATH}/lib/llvm/bin" "${ROCM_PATH}/bin")
endif()
foreach(_root IN LISTS _LDS_ROCM_ROOT_HINTS)
  list(APPEND _LDS_HIP_COMPILER_HINTS "${_root}/lib/llvm/bin" "${_root}/bin")
endforeach()
if(_LDS_HIP_COMPILER_HINTS)
  list(REMOVE_DUPLICATES _LDS_HIP_COMPILER_HINTS)
endif()

set(LDS_HIP_AVAILABLE FALSE)
if(LDS_BUILD_HIP_BENCHMARKS)
  set(_LDS_HIP_UNAVAILABLE_REASON "")

  if(NOT LDS_GPU_ARCH)
    message(FATAL_ERROR "LDS_GPU_ARCH is required when HIP benchmarks are enabled, e.g. -DLDS_GPU_ARCH=gfx942")
  endif()

  if(ROCM_PATH MATCHES "_rocm_sdk_core$")
    set(_LDS_HIP_UNAVAILABLE_REASON
      "TheRock runtime root was found at ${ROCM_PATH}, but native CMake HIP builds require initialized devel contents. Install rocm[devel], run `venv/bin/rocm-sdk init`, or set ROCM_PATH to a full ROCm SDK root.")
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND ROCM_PATH)
    lds_rocm_root_has_hip_language_package("${ROCM_PATH}" _has_hip_language_package)
    if(NOT _has_hip_language_package)
      set(_LDS_HIP_UNAVAILABLE_REASON
        "ROCm root ${ROCM_PATH} does not contain CMake's HIP language package.")
    endif()
  endif()

  set(_LDS_HIP_COMPILER "${CMAKE_HIP_COMPILER}")

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND _LDS_HIP_COMPILER AND NOT IS_ABSOLUTE "${_LDS_HIP_COMPILER}")
    find_program(_LDS_RESOLVED_HIP_COMPILER
      NAMES "${_LDS_HIP_COMPILER}"
      HINTS ${_LDS_HIP_COMPILER_HINTS}
      NO_CACHE
    )
    if(_LDS_RESOLVED_HIP_COMPILER)
      set(_LDS_HIP_COMPILER "${_LDS_RESOLVED_HIP_COMPILER}")
    endif()
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND _LDS_HIP_COMPILER)
    lds_normalize_hip_compiler("${_LDS_HIP_COMPILER}" _LDS_HIP_COMPILER)
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND _LDS_HIP_COMPILER AND IS_ABSOLUTE "${_LDS_HIP_COMPILER}" AND NOT EXISTS "${_LDS_HIP_COMPILER}")
    get_filename_component(_LDS_HIP_COMPILER_NAME "${_LDS_HIP_COMPILER}" NAME)
    find_program(_LDS_RESOLVED_MISSING_HIP_COMPILER
      NAMES "${_LDS_HIP_COMPILER_NAME}"
      HINTS ${_LDS_HIP_COMPILER_HINTS}
      NO_CACHE
    )
    if(_LDS_RESOLVED_MISSING_HIP_COMPILER)
      set(_LDS_HIP_COMPILER "${_LDS_RESOLVED_MISSING_HIP_COMPILER}")
    else()
      set(_LDS_HIP_UNAVAILABLE_REASON "CMAKE_HIP_COMPILER does not exist: ${_LDS_HIP_COMPILER}")
    endif()
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND _LDS_HIP_COMPILER)
    lds_normalize_hip_compiler("${_LDS_HIP_COMPILER}" _LDS_HIP_COMPILER)
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND NOT _LDS_HIP_COMPILER)
    find_program(_LDS_FOUND_HIP_COMPILER
      NAMES amdclang++ clang++
      HINTS ${_LDS_HIP_COMPILER_HINTS}
      NO_DEFAULT_PATH
      NO_CACHE
    )
    if(NOT _LDS_FOUND_HIP_COMPILER)
      find_program(_LDS_FOUND_HIP_COMPILER
        NAMES amdclang++
        NO_CACHE
      )
    endif()
    if(_LDS_FOUND_HIP_COMPILER)
      set(_LDS_HIP_COMPILER "${_LDS_FOUND_HIP_COMPILER}")
      lds_normalize_hip_compiler("${_LDS_HIP_COMPILER}" _LDS_HIP_COMPILER)
    endif()
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND _LDS_HIP_COMPILER)
    lds_hip_compiler_runs("${_LDS_HIP_COMPILER}" _LDS_HIP_COMPILER_RUNS _compiler_unavailable_reason)
    if(NOT _LDS_HIP_COMPILER_RUNS)
      set(_LDS_HIP_UNAVAILABLE_REASON "${_compiler_unavailable_reason}")
    endif()
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND NOT ROCM_PATH AND _LDS_HIP_COMPILER)
    lds_infer_rocm_root_from_compiler("${_LDS_HIP_COMPILER}" _LDS_INFERRED_ROCM_PATH)
    if(_LDS_INFERRED_ROCM_PATH)
      set(ROCM_PATH "${_LDS_INFERRED_ROCM_PATH}" CACHE PATH "ROCm SDK root" FORCE)
    endif()
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND ROCM_PATH)
    lds_rocm_root_has_hip_language_package("${ROCM_PATH}" _has_hip_language_package)
    if(NOT _has_hip_language_package)
      set(_LDS_HIP_UNAVAILABLE_REASON
        "ROCm root ${ROCM_PATH} does not contain CMake's HIP language package.")
    endif()
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND NOT _LDS_HIP_COMPILER)
    set(_LDS_HIP_UNAVAILABLE_REASON "A HIP compiler was not found.")
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON AND NOT ROCM_PATH)
    set(_LDS_HIP_UNAVAILABLE_REASON
      "A ROCm root containing CMake's HIP language package was not found.")
  endif()

  if(NOT _LDS_HIP_UNAVAILABLE_REASON)
    set(CMAKE_HIP_COMPILER "${_LDS_HIP_COMPILER}" CACHE FILEPATH "HIP compiler" FORCE)
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}" CACHE PATH "ROCm root for CMake HIP language support" FORCE)
    set(CMAKE_HIP_ARCHITECTURES "${LDS_GPU_ARCH}" CACHE STRING "HIP architectures" FORCE)

    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
      set(LDS_HIP_AVAILABLE TRUE)
    else()
      set(_LDS_HIP_UNAVAILABLE_REASON "CMake could not enable the HIP language with ${_LDS_HIP_COMPILER}.")
    endif()
  endif()

  if(_LDS_HIP_UNAVAILABLE_REASON AND LDS_REQUIRE_HIP)
    message(FATAL_ERROR "${_LDS_HIP_UNAVAILABLE_REASON}")
  elseif(_LDS_HIP_UNAVAILABLE_REASON)
    message(STATUS "${_LDS_HIP_UNAVAILABLE_REASON} HIP benchmark targets are disabled.")
  else()
    unset(_LDS_HIP_UNAVAILABLE_REASON)
  endif()
endif()

if(LDS_HIP_AVAILABLE AND ROCM_PATH AND NOT EXISTS "${ROCM_PATH}")
  message(FATAL_ERROR "ROCM_PATH does not exist: ${ROCM_PATH}")
endif()

if(LDS_HIP_AVAILABLE)
  message(STATUS "HIP benchmarks enabled")
  message(STATUS "  HIP compiler: ${CMAKE_HIP_COMPILER}")
  if(ROCM_PATH)
    message(STATUS "  ROCM_PATH: ${ROCM_PATH}")
  endif()
  message(STATUS "  LDS_GPU_ARCH: ${LDS_GPU_ARCH}")
endif()

function(add_hip_executable target)
  if(NOT LDS_HIP_AVAILABLE)
    message(FATAL_ERROR "add_hip_executable(${target}) called without an available HIP toolchain")
  endif()

  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "add_hip_executable(${target}) got unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "add_hip_executable(${target}) requires SOURCES")
  endif()

  set(abs_sources)
  foreach(src IN LISTS ARG_SOURCES)
    if(IS_ABSOLUTE "${src}")
      list(APPEND abs_sources "${src}")
    else()
      list(APPEND abs_sources "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endif()
  endforeach()

  set_source_files_properties(${abs_sources} PROPERTIES LANGUAGE HIP)
  add_executable("${target}" ${abs_sources})
  set_target_properties("${target}" PROPERTIES
    HIP_ARCHITECTURES "${LDS_GPU_ARCH}"
  )
  target_compile_options("${target}" PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--save-temps=obj>
    $<$<COMPILE_LANGUAGE:HIP>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-g>
  )

  if(ROCM_PATH AND EXISTS "${ROCM_PATH}/lib")
    set_target_properties("${target}" PROPERTIES
      BUILD_RPATH "${ROCM_PATH}/lib"
    )
  endif()
endfunction()

if(LDS_HIP_AVAILABLE)
  add_hip_executable(lds_phase_mask
    SOURCES
      bench/lds_phase_mask.hip.cpp
  )
endif()
