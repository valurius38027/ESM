function(sgw_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
      -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
      -Wunused -Woverloaded-virtual -Wnull-dereference
      -Wdouble-promotion -Wformat=2)
  endif()
endfunction()
