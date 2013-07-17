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

# Check whether to use the built-in libSimd
AC_ARG_WITH([built-in-simd],
    AS_HELP_STRING([--with-built-in-simd], [use statically linked embedded libSimd]), [
    SIMD_LIBTOOL="\$(top_builddir)/simd/src/libSimd.la"
    SIMD_DIR=simd
    AC_CONFIG_SUBDIRS([simd])
    CPPFLAGS="$CPPFLAGS -I\$(top_srcdir)/simd/inc"    
], [
    # Check for installed SIMD
    PKG_CHECK_MODULES([SIMD], [Simd >= 1.0])
])
AC_SUBST([SIMD_LIBTOOL])
AC_SUBST([SIMD_DIR])

AC_OUTPUT
