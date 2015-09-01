#  This file is a part of SEAPT, Samsung Extended Autotools Project Template

#  Copyright 2012,2013 Samsung R&D Institute Russia
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met: 
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer. 
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

AC_INIT(Znicz, 1.0)
m4_include([common.ac])
AC_SUBST(INTERFACE_VERSION, [0])
AC_SUBST(REVISION_NUMBER, [$(cd $srcdir && git rev-list HEAD --count)])
AC_SUBST(AGE_NUMBER, [0])

# Get libVeles headers
AC_ARG_WITH([veles-headers],
    AS_HELP_STRING([--with-veles-headers], [take Veles include files from this path]), [
    AS_IF([test $(echo "$with_veles_headers" | cut -c1) != '/'], [
        VELES_CFLAGS="-I\$(abs_top_builddir)/$with_veles_headers"
    ], [
        VELES_CFLAGS="-I$with_veles_headers"
    ]) 
])

# Get libSimd headers
AC_ARG_WITH([simd-headers],
    AS_HELP_STRING([--with-simd-headers], [take libSimd include files from this path]), [
    AS_IF([test $(echo "$with_simd_headers" | cut -c1) != '/'], [
        SIMD_CFLAGS="-I\$(abs_top_builddir)/$with_simd_headers"
    ], [
        SIMD_CFLAGS="-I$with_simd_headers"
    ])
])

# Get libVeles build root
AC_ARG_WITH([veles-build-root],
    AS_HELP_STRING([--with-veles-build-root], [link with Veles in this path]), [
    AS_IF([test $(echo "$with_veles_build_root" | cut -c1) != '/' -a "x$with_veles_build_root" != "xno"], [
        VELES_LIBS="\$(abs_top_builddir)/$with_veles_build_root/src/libVeles.la \
        \$(abs_top_builddir)/$with_veles_build_root/simd/src/libSimd.la"
    ], [
        AS_IF([test "x$with_veles_build_root" != "xno"], [
            VELES_LIBS="$with_veles_build_root/src/libVeles.la \
            $with_veles_build_root/simd/src/libSimd.la"
        ])
    ])
])

AS_IF([test "x$with_veles_headers" = "x" -o "x$with_veles_build_root" = "x"], [
    # Check for installed libVeles
    PKG_CHECK_MODULES([VELES], [Veles >= 1.0])
])

CPPFLAGS="$CPPFLAGS $VELES_CFLAGS $SIMD_CFLAGS"

# Check whether to use nice Eina logging
AC_ARG_ENABLE([eina-logging],
    AS_HELP_STRING([--disable-eina-logging], [do not use Eina as the logging backend])
)
AC_ARG_WITH([built-in-eina-logging],
AS_HELP_STRING([--with-built-in-eina-logging], [use bundled eina_log module]), [
    AS_IF([test "x$enable_eina_logging" != "xno"], [
        EINA_CFLAGS="-DEINA_STRIPPED -I\$(top_srcdir)/eina"
        AS_IF([test "x$with_veles_build_root" != "xno" -a "x$with_veles_build_root" != "xno"], [
            AS_IF([test $(echo "$with_veles_build_root" | cut -c1) != '/'], [
                VELES_LIBS="$VELES_LIBS \$(abs_top_builddir)/$with_veles_build_root/eina/libEina.la"
            ], [
                AS_IF([test "x$with_veles_build_root" != "xno"], [
                    VELES_LIBS="$VELES_LIBS $with_veles_build_root/eina/libEina.la"
                ])
            ])
        ])
    ])
])
AS_IF([test "x$enable_eina_logging" != "xno"], [
    AM_CPPFLAGS="$AM_CPPFLAGS -DEINA $EINA_CFLAGS"
])

AC_CONFIG_COMMANDS([Creating workflow_files directory in tests], 
    [mkdir -p tests/workflow_files])

AM_COND_IF([TESTS], [
    AC_CONFIG_LINKS([tests/workflow_files/mnist.zip:tests/workflow_files/mnist.zip])
])

AC_OUTPUT

AC_DEFUN([ZNICZ_PRINT_STATUS], [
echo
echo -e "${COLOR_WHITE}$PACKAGE_NAME options:${COLOR_RESET}"
echo -e "  veles_headers......: ${COLOR_CYAN}${with_veles_headers:-no}${COLOR_RESET}"
echo -e "  simd_headers......: ${COLOR_CYAN}${with_simd_headers:-no}${COLOR_RESET}"
echo -e "  veles_build_root...: ${COLOR_CYAN}${with_veles_build_root:-no}${COLOR_RESET}"
echo -e "  eina_logging............: $(color_yes_no ${enable_eina_logging:-yes})"
echo -e "  built_in_eina_logging...: $(color_yes_no ${with_built_in_eina_logging:-no})"
echo
])

COMMON_PRINT_STATUS
ZNICZ_PRINT_STATUS

