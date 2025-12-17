# Quasistatic Data for SoPrA

We store the CSV files for the data in: https://www.dropbox.com/scl/fo/cwaj8mcfinkzb732773jd/AOsO8sQDe1n3YYfnvmywbF4?rlkey=d0qyis7y0pbjkndexxa11bd1q&st=0ygaaz53&dl=0

Please download and store those CSV files directly here.


## Data Explanation

seg_1_angle_1
seg_1_angle_2
seg_2_angle_1
seg_2_angle_2

I think this was measured from Bendlabs, but you can probably just ignore these anyways.

pressure_0,pressure_1,pressure_2,pressure_3,pressure_4,pressure_5 --> pressure applied to the 6 chambers

xm_WB,ym_WB,zm_WB --> position World to Base (measured, from qualisys)
qwm_WB,qxm_WB,qym_WB,qzm_WB --> quaternion World to Base (measured, from qualisys)

then there's World to Intermediate, and World to Tip

xm_WB,ym_WB,zm_WB,qwm_WB,qxm_WB,qym_WB,qzm_WB,
xm_WI,ym_WI,zm_WI,qwm_WI,qxm_WI,qym_WI,qzm_WI,
xm_WT,ym_WT,zm_WT,qwm_WT,qxm_WT,qym_WT,qzm_WT,

xe_BI,ye_BI,ze_BI  --> position Base to Intermediate (estimated, in SOFA)
qwe_BI,qxe_BI,qye_BI,qze_BI --> quaternion Base to Intermediate (estimated, in SOFA)

then there's Base to Tip

xe_BI,ye_BI,ze_BI,qwe_BI,qxe_BI,qye_BI,qze_BI,
xe_BT,ye_BT,ze_BT,qwe_BT,qxe_BT,qye_BT,qze_BT