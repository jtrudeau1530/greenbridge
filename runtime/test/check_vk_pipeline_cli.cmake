file(MAKE_DIRECTORY "${TEST_DIR}")
file(WRITE "${TEST_DIR}/index.json" "not JSON\n")

# ${RUNNER} is CMAKE_CROSSCOMPILING_EMULATOR: CTest does not apply it to a script
# test (the command is the host CMake), so the script runs the tool through it itself.
function(check_invalid option)
  execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}" "${option}" ${ARGN}
                  RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
  if(NOT code STREQUAL "2" OR NOT log MATCHES "${option} requires a valid value")
    message(FATAL_ERROR "${option} ${ARGN}: expected usage error, got ${code}\n${log}")
  endif()
endfunction()

check_invalid(--count)
check_invalid(--count --compute-only)
check_invalid(--count 18446744073709551616)
check_invalid(--jobs 4294967296)
check_invalid(--start -1)
check_invalid(--max-bytes 1MiB)
check_invalid(--large-module-bytes -1)
check_invalid(--slow nonsense)
check_invalid(--slow nan)
check_invalid(--slow -0.5)
check_invalid(--stage --graphics-only)
check_invalid(--stats)

execute_process(COMMAND ${RUNNER} "${TOOL}" "${TEST_DIR}"
                --jobs 2 --start 3 --count 0 --max-bytes 0 --large-module-bytes 0
                --slow 0.5 --stage cs --stats report.csv
                RESULT_VARIABLE code OUTPUT_VARIABLE log ERROR_VARIABLE log TIMEOUT 5)
if(NOT code STREQUAL "2" OR NOT log MATCHES "malformed JSON")
  message(FATAL_ERROR "Expected valid arguments to reach the malformed index, got ${code}\n${log}")
endif()
