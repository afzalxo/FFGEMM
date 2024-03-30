<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" top="uut_top" name="gemm_test.prj">
    <includePaths/>
    <libraryFlag/>
    <files>
        <file name="../../test.cpp" sc="0" tb="1" cflags=" -I../../../../../include/hw  -I../../../../sw/include  -I../../../gemm  -I../../../../../include/hw/xf_blas/helpers/utils -std=c++11  -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../../../..//L1/tests/hw/gemm/uut_top.cpp" sc="0" tb="false" cflags="-I../../../..//L1/include/hw -I../../../..//L1/tests/hw/gemm -I../../../..//L1/include/hw/xf_blas/helpers/utils -std=c++11" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="sol" status=""/>
    </solutions>
    <Simulation argv="/home/x2/afzal/blocked_systolic_array/src/L1/tests/hw/gemm/data/">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

