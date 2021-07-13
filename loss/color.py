import tensorflow as tf
import sys


'''transferring a nhw3 img in RGB to nhw3 XYZ'''
def rgb2xyz(img, gamut='709'):
    img_r = tf.slice(img, [0, 0, 0, 0], [-1, -1, -1, 1])
    img_g = tf.slice(img, [0, 0, 0, 1], [-1, -1, -1, 1])
    img_b = tf.slice(img, [0, 0, 0, 2], [-1, -1, -1, 1])
    # all these matrices below are assumed to D65 reference white (excluding 601 for C white)
    if gamut == '601':
        img_x = 0.607 * img_r + 0.174 * img_g + 0.200 * img_b
        img_y = 0.299 * img_r + 0.587 * img_g + 0.114 * img_b
        img_z = 0.000 * img_r + 0.066 * img_g + 1.116 * img_b
    elif gamut == '709':
        img_x = 0.4124 * img_r + 0.3576 * img_g + 0.1805 * img_b
        img_y = 0.2126 * img_r + 0.7152 * img_g + 0.0722 * img_b
        img_z = 0.0193 * img_r + 0.1192 * img_g + 0.9505 * img_b
    elif gamut == '2020':
        img_x = 0.6370 * img_r + 0.1446 * img_g + 0.1689 * img_b
        img_y = 0.2627 * img_r + 0.6780 * img_g + 0.0593 * img_b
        img_z = 0.0000 * img_r + 0.0281 * img_g + 1.0610 * img_b
    elif gamut == 'DCI-P3':
        img_x = 0.4866 * img_r + 0.2657 * img_g + 0.1892 * img_b
        img_y = 0.2290 * img_r + 0.6917 * img_g + 0.0793 * img_b
        img_z = 0.0000 * img_r + 0.0451 * img_g + 1.0439 * img_b
    elif gamut == 'Alexa':
        img_x = 0.638008 * img_r + 0.214704 * img_g + 0.097744 * img_b
        img_y = 0.291954 * img_r + 0.823841 * img_g - 0.115795 * img_b
        img_z = 0.002798 * img_r - 0.067034 * img_g + 1.153294 * img_b
    elif gamut == 'Fairchild Nikon D2x':
        img_x = 0.4024 * img_r + 0.4610 * img_g + 0.0871 * img_b
        img_y = 0.1904 * img_r + 0.7646 * img_g + 0.0450 * img_b
        img_z = -0.0249 * img_r + 0.1264 * img_g + 0.9873 * img_b
    else:
        sys.exit('Input gamut is not supported yet.')

    # return img_x, img_y, img_z
    return tf.maximum(1e-6, img_x), tf.maximum(1e-6, img_y), tf.maximum(1e-6, img_z)
    # output is truncated to avoid 0 which will cause loss NaN (?)


'''transferring a nhw3 XYZ to nhw3 Lab'''
def xyz2lab(img_x, img_y, img_z):
    # we used simplified ver. of f(t)=t ** 1/3, & E reference white (X0=Y0=Z0=1) for computational simplicity
    img_l = 116 * (img_y ** (1/3)) - 16
    img_a = 500 * (img_x ** (1/3) - img_y ** (1/3))
    img_b = 200 * (img_x ** (1/3) - img_z ** (1/3))
    '''TODO: separating a & b to pos. / neg. 2 parts to avoid 0'''
    return img_l, img_a, img_b


'''transferring a nhw3 XYZ to nhw3 LMS'''
def xyz2lms(img_x, img_y, img_z, next_step):
    if next_step == 'IPT':
        img_l = 0.4002 * img_x + 0.7076 * img_y - 0.0808 * img_z
        img_m = -0.2263 * img_x + 1.1653 * img_y + 0.0457 * img_z
        img_s = 0.0000 * img_x + 0.0000 * img_y + 0.9182 * img_z
    elif next_step == 'ICtCp':
        # cross-talked matrix specialized for ICtCp (see Dolby ICtCp White Paper)
        img_l = 0.3592 * img_x + 0.6976 * img_y - 0.0358 * img_z
        img_m = -0.1922 * img_x + 1.1004 * img_y + 0.0755 * img_z
        img_s = 0.0070 * img_x + 0.0749 * img_y + 0.8434 * img_z
    else:
        sys.exit('Input next_step is not supported yet.')
    # transferring L, M, S to non-linear L', M', S' using default 0.43 rather than PQ/HLG/gamma OETF
    return tf.maximum(1e-6, img_l ** 0.43), tf.maximum(1e-6, img_m ** 0.43), tf.maximum(1e-6, img_s ** 0.43)
    # output is truncated to avoid 0 which will cause loss NaN (?)


'''transferring a nhw3 LMS to nhw3 IPT'''
def lms2ipt(img_l, img_m, img_s):
    img_i = 0.4 * img_l + 0.4 * img_m + 0.2 * img_s
    img_p = 4.4550 * img_l - 4.8510 * img_m + 0.3960 * img_s
    img_t = 0.8056 * img_l + 0.3572 * img_m - 1.1628 * img_s
    '''TODO: separating a & b to pos. / neg. 2 parts to avoid 0'''
    return img_i, img_p, img_t


'''transferring a nhw3 LMS to nhw3 ICtCp'''
def lms2ictcp(img_l, img_m, img_s):
    img_i = 0.5 * img_l + 0.5 * img_m + 0.0 * img_s
    img_ct = 1.6137 * img_l - 3.3234 * img_m + 1.7097 * img_s
    img_cp = 4.3780 * img_l - 4.2455 * img_m - 0.1325 * img_s
    '''TODO: separating a & b to pos. / neg. 2 parts to avoid 0'''
    return img_i, img_ct, img_cp


'''BT.2124 deltaEitp color difference in ICtCp color space'''
def delta_e_itp(img_1, img_2, gamut1='709', gamut2='709'):
    img_x1, img_y1, img_z1 = rgb2xyz(img_1, gamut=gamut1)
    img_x2, img_y2, img_z2 = rgb2xyz(img_2, gamut=gamut2)
    img_l1, img_m1, img_s1 = xyz2lms(img_x1, img_y1, img_z1, next_step='ICtCp')
    img_l2, img_m2, img_s2 = xyz2lms(img_x2, img_y2, img_z2, next_step='ICtCp')
    img_i1, img_ct1, img_cp1 = lms2ictcp(img_l1, img_m1, img_s1)
    img_i2, img_ct2, img_cp2 = lms2ictcp(img_l2, img_m2, img_s2)
    delta_e_itp_map = tf.sqrt(tf.square(img_i1 - img_i2) + tf.square(0.5 * img_ct1 - 0.5 * img_ct2) + tf.square(img_cp1 - img_cp2))
    return tf.reduce_mean(delta_e_itp_map)


'''hue difference in IPT (ICh) color space'''
def delta_hue_ipt(img_1, img_2, gamut1='709', gamut2='709'):
    img_x1, img_y1, img_z1 = rgb2xyz(img_1, gamut=gamut1)
    img_x2, img_y2, img_z2 = rgb2xyz(img_2, gamut=gamut2)
    img_l1, img_m1, img_s1 = xyz2lms(img_x1, img_y1, img_z1, next_step='IPT')
    img_l2, img_m2, img_s2 = xyz2lms(img_x2, img_y2, img_z2, next_step='IPT')
    _, img_p1, img_t1 = lms2ipt(img_l1, img_m1, img_s1)
    _, img_p2, img_t2 = lms2ipt(img_l2, img_m2, img_s2)
    delta_hue_ipt_map = tf.abs(tf.atan(img_p1 / (img_t1 + 1e-6)) - tf.atan(img_p2 / (img_t2 + 1e-6)))
    return tf.reduce_mean(delta_hue_ipt_map)

'''deltaE76 color & hue difference in Lab (LCh) color space'''
def delta_e_76(img_1, img_2, gamut1='709', gamut2='709'):
    img_x1, img_y1, img_z1 = rgb2xyz(img_1, gamut=gamut1)
    img_x2, img_y2, img_z2 = rgb2xyz(img_2, gamut=gamut2)
    img_l1, img_a1, img_b1 = xyz2lab(img_x1, img_y1, img_z1)
    img_l2, img_a2, img_b2 = xyz2lab(img_x2, img_y2, img_z2)
    delta_e_76_map = tf.sqrt(tf.square(img_l1 - img_l2) + tf.square(img_a1 - img_a2) + tf.square(img_b1 - img_b2))
    delta_hue_76_map = tf.abs(tf.atan(img_b1 / (img_a1 + 1e-6)) - tf.atan(img_b2 / (img_a2 + 1e-6)))
    return tf.reduce_mean(delta_e_76_map), tf.reduce_mean(delta_hue_76_map)