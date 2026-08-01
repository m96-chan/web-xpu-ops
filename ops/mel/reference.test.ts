import { describe, expect, it } from "vitest";
import { agree } from "../../harness/index.js";
import { hannWindow, stft } from "../stft/reference.js";
import {
  hzToMel,
  melFilterbank,
  melFilterbankHasEmptyBand,
  melFrequencies,
  melSpectrogram,
  melToHz,
} from "./reference.js";

/**
 * The conventions, pinned to the libraries they claim to match.
 *
 * Every number below was produced by **torchaudio 2.10.0+cu128** or
 * **librosa 0.11.0** and pasted in. This is the only test in this op that can
 * catch a convention being wrong rather than an arithmetic slip: the WGSL tests
 * compare a kernel against this reference, so if the reference picked the
 * Slaney mel scale by default, or normalised the triangles, or took the log of
 * a magnitude spectrum with the power multiplier, the kernel and the reference
 * would agree with each other all the way to a wrong feature.
 *
 * The audio is real — 512 samples of `/usr/share/sounds/alsa/Front_Center.wav`,
 * a recorded human voice, taken at the loudest point of the utterance (offset
 * 47476, RMS 0.218) and pasted here as the int16 the file stores. It is divided
 * by 32768 exactly as `torchaudio.load` does, so torch and this reference are
 * looking at the same samples rather than at two renderings of a formula.
 *
 * Touches no GPU.
 */

// ---------------------------------------------------------------- tolerances

/**
 * Tolerances are measured, not chosen. Each was read off a run of this file
 * before it was written down.
 *
 * `SCALE` — `hzToMel`, `melToHz` and `melFrequencies` are f64 here and f64 in
 * torchaudio and librosa, and the arithmetic is in the same order, so the
 * measured difference over every probe below is **exactly zero, bit for bit**.
 * The tolerance is not slack for a known error; it is there so that a future
 * refactor moving one multiply does not fail on an ulp.
 *
 * `FILTERBANK` — librosa builds the matrix in f64 and stores f32, which is what
 * this does, so the gap is a single f32 rounding: worst observed **2.7e-8
 * absolute** on a bank peaking at 1. The `abs` leg is scaled to that peak, and
 * it is also what passes the entries that should be exactly zero and come out
 * of librosa at 5e-16 — the top edge of the last triangle.
 * `FILTERBANK_NORMALISED` is the same 1e-7-of-peak rule for a Slaney-normalised
 * bank, whose peak is 1.3e-3 rather than 1: worst observed 4.2e-11 absolute.
 * Reusing the unnormalised `abs` there would have accepted a 0.3% error on the
 * smallest weights.
 *
 * `MEL` — torchaudio runs the whole chain in f32: its bin frequencies, its
 * filterbank, its STFT and its matmul. This reference runs in f64 and rounds
 * twice, once into the spectrogram and once into the output. The disagreement
 * is therefore torch's accumulated f32 error rather than this one's, and it is
 * concentrated exactly where that theory predicts — measured per mel band, it
 * runs 1.4e-7 relative in band 0 and climbs monotonically to **4.6e-5 in band
 * 15**, which is the quietest band: a 129-term f32 dot product over values
 * spanning seven decades has nothing left by the time it reaches 4.5e-5. That
 * was confirmed against a float64 rerun of torchaudio's own code path, which
 * agrees with this reference to 4.0e-5 in the same place. Worst over the three
 * pinned linear spectrograms: 4.61e-5.
 *
 * `MEL_DB` — the same error seen through `10 log10`, plus `20 log10` for the
 * magnitude flavour, which doubles it. Worst observed **4.0e-4 absolute**. The
 * `abs` leg carries this one because dB crosses zero — two of the pinned values
 * are 0.086 and -0.079 — and relative error there means nothing.
 *
 * Every convention this file guards against is wrong by a factor, not by a
 * rounding: htk against slaney is 200x on the same audio, power against
 * magnitude is 20x, so 4.6e-5 is three orders clear of anything it must catch.
 */
const SCALE = { rel: 1e-14, abs: 1e-12 };
const FILTERBANK = { rel: 1e-6, abs: 1e-7 };
const FILTERBANK_NORMALISED = { rel: 1e-6, abs: 1e-10 };
const MEL = { rel: 6e-5, abs: 1e-9 };
const MEL_DB = { rel: 1e-5, abs: 5e-4 };

// ------------------------------------------------------- pinned: the mel scale

const PROBE_HZ = [
  0.0, 100.0, 700.0, 999.9, 1000.0, 1000.1,
  4000.0, 8000.0, 24000.0,
];
// torchaudio.functional.functional._hz_to_mel(f, "htk")
const PROBE_MEL_HTK = [
  0.0, 150.48910240709708, 781.1728387480312, 999.9192414144051, 999.9855371396244, 1000.0518289652098,
  2146.06452750619, 2840.023046708319, 4016.019179871836,
];
// ..._hz_to_mel(f, "slaney"). Linear at 200/3 Hz per mel until 1 kHz — note
// 100 Hz -> 1.5 and 700 Hz -> 10.5 exactly — then logarithmic.
const PROBE_MEL_SLANEY = [
  0.0, 1.5, 10.5, 14.998499999999998, 14.999999999999998, 15.001454435130032,
  35.163760314616646, 45.245640471924965, 61.225042458023374,
];

// librosa.mel_frequencies(n_mels=10, fmin=0, fmax=8000, htk=...)
const EDGES_HTK = [
  0.0, 226.19065968749445, 525.4701972747943, 921.4557863447225, 1445.3960062981764, 2138.6367748060343,
  3055.8840958154033, 4269.52109773291, 5875.3200340573785, 8000.000000000002,
];
const EDGES_SLANEY = [
  0.0, 335.1528923846294, 670.3057847692588, 1005.6452812331256, 1420.863467108422, 2007.5199773103307,
  2836.3995222579288, 4007.5129218110633, 5662.164195295692, 7999.999999999998,
];

// ------------------------------------------------------ pinned: the filterbank

const FB_SR = 16000;
const FB_NFFT = 32;
const FB_BINS = 17;
const FB_NMELS = 4;

// librosa.filters.mel(sr=16000, n_fft=32, n_mels=4, htk=True, norm=None),
// which is also torchaudio.functional.melscale_fbanks(..., norm=None,
// mel_scale="htk").T — the two agree to 3.9e-7, torchaudio's f32 arithmetic.
// This is the default here: triangles that peak at exactly 1.
const FB_HTK_NONE = [
  -0.0, 0.9456509291745406, 0.2871922301763404, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.05434907082545936, 0.7128077698236597, 0.7757143515670198, 0.37793311174811073, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.2242856484329801, 0.6220668882518893, 0.988009554731225, 0.7477060778501317, 0.5074026009690384,
  0.26709912408794506, 0.02679564720685178, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.011990445268774992, 0.2522939221498683, 0.4925973990309616, 0.7329008759120549,
  0.9732043527931482, 0.8710178644035251, 0.7258482203362709, 0.5806785762690169, 0.4355089322017628, 0.2903392881345087,
  0.14516964406725463, 5.281240885497176e-16,
];

// librosa.filters.mel(sr=16000, n_fft=32, n_mels=4) — its own defaults, i.e.
// htk=False and norm="slaney". Three orders of magnitude smaller than the row
// above and shaped differently: this is what "pick a convention silently" costs.
const FB_SLANEY_SLANEY = [
  -0.0, 0.0013396001465150744, 0.0006050957639020401, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0007352135534891176, 0.0008861480886786564, 0.0003358614972034313, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0001609024541715258, 0.00046726403337509203, 0.0005901677984859587, 0.00042571519645100484, 0.0002612625944160509,
  9.680999238109699e-05, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 3.439805434922838e-05, 0.00012267497989052392, 0.00021095190543181944, 0.000299228830973115,
  0.0003317047490046382, 0.00028431835628968985, 0.00023693196357474152, 0.0001895455708597932, 0.00014215917814484484, 9.477278542989652e-05,
  4.738639271494817e-05, 0.0,
];

// The two crossed corners, one band each, to show `scale` and `norm` are
// separate axes rather than one preset with two names.
const FB_ROW2_HTK_SLANEY = [
  0.0, 0.0, 0.0, 0.00013439635343220246, 0.00037275466333260856, 0.0005920346764930383,
  0.00044804012652722086, 0.0003040455765614035, 0.00016005102659558606, 1.6056476629768687e-05, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0,
];
const FB_ROW2_SLANEY_NONE = [
  0.0, 0.0, 0.0, 0.24593476021522195, 0.7141995974952443, 0.9020544574954964,
  0.6506933986696267, 0.3993323398437569, 0.14797128101788715, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0,
];

// ------------------------------------------------------- pinned: on real audio

const A_SR = 48000;
const A_NFFT = 256;
const A_HOP = 128;
const A_MELS = 16;
const A_FRAMES = 5;
const A_BINS = 129;

/** int16 samples 47476..47987 of `/usr/share/sounds/alsa/Front_Center.wav`. */
const CLIP = [
  -7465, -7538, -7614, -7880, -8363, -8805, -9027, -9249, -9650, -10072, -10324, -10424, -10598, -11005,
  -11459, -11765, -12087, -12581, -13064, -13393, -13693, -14055, -14384, -14629, -14768, -14535, -13672, -12329,
  -10724, -8787, -6480, -4165, -2343, -1144, -390, 32, 89, -245, -774, -1203, -1422, -1406,
  -1218, -1127, -1291, -1514, -1668, -1950, -2367, -2601, -2562, -2438, -2301, -2004, -1483, -1047,
  -1007, -1108, -902, -540, -338, -185, 17, 358, 723, 546, -75, -773, -1796, -3041,
  -4246, -5396, -6242, -6786, -7128, -7042, -6550, -5897, -5248, -4581, -3760, -2882, -2064, -1254,
  -437, 465, 1441, 2337, 3257, 4210, 4926, 5544, 6306, 7053, 7645, 8111, 8442, 8630,
  8716, 8727, 8624, 8406, 8221, 8169, 8211, 8352, 8649, 9090, 9641, 10304, 11062, 11777,
  12331, 12754, 13061, 13288, 13448, 13317, 12802, 12109, 11372, 10615, 9833, 9007, 8203, 7451,
  6766, 6231, 5749, 5122, 4428, 3874, 3467, 3101, 2740, 2432, 2311, 2491, 2871, 3216,
  3424, 3579, 3849, 4256, 4539, 4538, 4464, 4443, 4333, 4080, 3729, 3331, 2953, 2596,
  2352, 2359, 2492, 2635, 2902, 3304, 3723, 4059, 4236, 4293, 4308, 4312, 4341, 4374,
  4323, 4186, 4085, 4030, 3819, 3372, 2828, 2278, 1716, 1111, 457, -233, -961, -1682,
  -2362, -3071, -3835, -4604, -5279, -5777, -6156, -6415, -6540, -6672, -6829, -6921, -6993, -7176,
  -7599, -8176, -8649, -9039, -9514, -10056, -10576, -10989, -11255, -11559, -12041, -12447, -12596, -12729,
  -13006, -13361, -13661, -13799, -13996, -14447, -14856, -14826, -14198, -13049, -11587, -9879, -7865, -5674,
  -3737, -2483, -1907, -1676, -1655, -1974, -2572, -3060, -3148, -2886, -2443, -1985, -1725, -1819,
  -2171, -2503, -2735, -2992, -3263, -3332, -3002, -2398, -1892, -1601, -1404, -1282, -1245, -1223,
  -1151, -971, -654, -272, 40, 198, 122, -369, -1373, -2636, -3777, -4652, -5393, -6100,
  -6476, -6228, -5608, -5009, -4557, -4137, -3526, -2777, -2173, -1725, -1145, -292, 687, 1655,
  2594, 3496, 4363, 5164, 5865, 6510, 7062, 7403, 7565, 7705, 7893, 8014, 7984, 7908,
  7901, 7937, 7988, 8104, 8273, 8460, 8818, 9458, 10216, 10927, 11580, 12219, 12826, 13230,
  13282, 13067, 12703, 12193, 11632, 11179, 10780, 10329, 9859, 9423, 9007, 8512, 7803, 6928,
  6121, 5486, 4946, 4449, 4056, 3891, 3976, 4142, 4247, 4297, 4348, 4445, 4587, 4708,
  4769, 4825, 4914, 4958, 4838, 4525, 4109, 3681, 3231, 2784, 2467, 2293, 2195, 2201,
  2349, 2541, 2644, 2652, 2617, 2557, 2523, 2517, 2504, 2550, 2715, 2941, 3133, 3222,
  3192, 3079, 2892, 2576, 2116, 1534, 870, 205, -469, -1174, -1869, -2572, -3269, -3914,
  -4510, -5006, -5372, -5673, -5965, -6269, -6539, -6756, -7021, -7389, -7797, -8208, -8587, -8964,
  -9467, -10029, -10467, -10852, -11285, -11768, -12360, -12941, -13342, -13720, -14219, -14707, -15105, -15411,
  -15487, -15200, -14525, -13415, -11987, -10493, -8920, -7328, -6134, -5457, -5071, -4956, -5065, -5178,
  -5183, -4922, -4231, -3338, -2574, -2036, -1843, -2045, -2442, -2920, -3414, -3712, -3768, -3620,
  -3118, -2381, -1832, -1587, -1498, -1449, -1380, -1364, -1389, -1178, -636, -69, 217, 179,
  -119, -693, -1567, -2633, -3603, -4219, -4567, -4806, -4839, -4628, -4335, -4121, -4101, -4180,
  -4106, -3857, -3568, -3187, -2520, -1528, -410, 632, 1570, 2523, 3519, 4355, 4948, 5469,
  5961, 6307, 6490, 6643, 6906, 7181, 7302, 7365, 7522, 7736, 7923, 8049, 8155, 8404,
  8883, 9454, 10020, 10655, 11313, 11885, 12333, 12560, 12578, 12497, 12301, 12081, 11997, 11985,
  11924, 11791, 11547, 11140, 10566, 9811, 8906, 8008,
];

// torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=256,
//   hop_length=128, n_mels=16, power=2.0, norm=None, mel_scale="htk")(clip),
// transposed to [frames, n_mels]. Everything else is that transform's default:
// centred, reflect-padded, one-sided, unnormalised, periodic Hann.
const MEL_POWER_HTK = [
  325.5381164550781, 146.63134765625, 24.635465621948242, 51.91167068481445, 43.60346984863281, 23.257896423339844,
  8.8213472366333, 5.760949611663818, 1.434881567955017, 0.4506025016307831, 0.007972629740834236, 0.012843596749007702,
  0.04502154514193535, 0.015393524430692196, 0.0016018184833228588, 0.0002656489086803049, 193.27200317382812, 167.10939025878906,
  52.633811950683594, 47.39597702026367, 3.936326742172241, 17.346187591552734, 2.347100257873535, 1.5731723308563232,
  0.19948451220989227, 0.08371610194444656, 0.0034888070076704025, 0.020382393151521683, 0.018090402707457542, 0.004987876396626234,
  0.0011364935198798776, 0.0002114949020324275, 216.60702514648438, 151.17800903320312, 57.248966217041016, 53.21343994140625,
  6.363376140594482, 23.524045944213867, 2.8862967491149902, 3.234584093093872, 1.989029884338379, 0.5773278474807739,
  0.00449019018560648, 0.031666647642850876, 0.04677211120724678, 0.006999032571911812, 0.0004617169906850904, 4.784454722539522e-05,
  221.75498962402344, 211.6431121826172, 65.00758361816406, 39.912235260009766, 3.9300308227539062, 12.193497657775879,
  1.869478464126587, 1.3533174991607666, 1.4051967859268188, 0.3828180730342865, 0.004696296062320471, 0.008262228220701218,
  0.010910536162555218, 0.0022603804245591164, 0.0013197566149756312, 4.4788488594349474e-05, 330.55169677734375, 272.31201171875,
  41.304283142089844, 19.291471481323242, 4.004454135894775, 10.13190746307373, 3.92459774017334, 0.29542234539985657,
  0.9346997141838074, 0.73578941822052, 0.09012474119663239, 0.038001593202352524, 0.039886508136987686, 0.014932295307517052,
  0.00788669753819704, 0.004472235217690468,
];

// torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None) of the above.
const MEL_POWER_HTK_DB = [
  25.126018524169922, 21.662269592285156, 13.915607452392578, 17.152650833129883, 16.39521026611328, 13.665704727172852,
  9.45534896850586, 7.604940414428711, 1.5681605339050293, -3.462064027786255, -20.983983993530273, -18.91313362121582,
  -13.465795516967773, -18.126619338989258, -27.953868865966797, -35.756919860839844, 22.861690521240234, 22.23000717163086,
  17.212648391723633, 16.757413864135742, 5.950911521911621, 12.392040252685547, 3.7053165435791016, 1.9677629470825195,
  -7.000908374786377, -10.771909713745117, -24.573230743408203, -16.907447814941406, -17.425518035888672, -23.020841598510742,
  -29.4443302154541, -36.74700164794922, 23.356725692749023, 21.79488754272461, 17.57767677307129, 17.260211944580078,
  8.03687572479248, 13.715120315551758, 4.603409767150879, 5.098184585571289, 2.98641300201416, -2.385775089263916,
  -23.477352142333984, -14.993978500366211, -13.300130844116211, -21.54962158203125, -33.35624313354492, -43.20167541503906,
  23.45873260498047, 23.25604248046875, 18.129640579223633, 16.01106071472168, 5.9439592361450195, 10.861283302307129,
  2.7172045707702637, 1.3139969110488892, 1.4773714542388916, -4.170075416564941, -23.28244400024414, -20.829029083251953,
  -19.621540069580078, -26.45818519592285, -28.795061111450195, -43.48833465576172, 25.192394256591797, 24.350666046142578,
  16.159950256347656, 12.853652954101562, 6.02543306350708, 10.05691146850586, 5.93795108795166, -5.295566558837891,
  -0.29327890276908875, -1.3324646949768066, -10.451560020446777, -14.201982498168945, -13.991739273071289, -18.25873374938965,
  -21.031047821044922, -23.494754791259766,
];

// The same transform with power=1.0. Not a rescaling of the power feature: the
// filterbank sums |X| here and |X|² above, and a sum of roots is not the root
// of a sum. Compare element 0: 16.92 against sqrt(325.54) = 18.04.
const MEL_MAGNITUDE_HTK = [
  16.920795440673828, 12.96725845336914, 4.969566822052002, 8.955816268920898, 9.572395324707031, 6.835385799407959,
  3.977247953414917, 4.085267066955566, 2.506211519241333, 1.0176870822906494, 0.2013811320066452, 0.28083837032318115,
  0.6001629829406738, 0.3459728956222534, 0.14292563498020172, 0.062233779579401016, 13.037799835205078, 13.981322288513184,
  7.744818687438965, 8.027220726013184, 2.34043550491333, 6.061076641082764, 1.756245732307434, 2.0045762062072754,
  0.9630479216575623, 0.4345441162586212, 0.14767558872699738, 0.31325262784957886, 0.39366430044174194, 0.22412630915641785,
  0.1280250996351242, 0.03722985088825226, 13.802447319030762, 13.301714897155762, 8.169811248779297, 8.563613891601562,
  2.7478859424591064, 7.194624900817871, 2.295442819595337, 3.296035051345825, 3.0129950046539307, 1.1357697248458862,
  0.16960786283016205, 0.4323381781578064, 0.6651158332824707, 0.22538158297538757, 0.0795048326253891, 0.019644755870103836,
  13.96550178527832, 16.02204132080078, 9.306180953979492, 7.594645023345947, 2.6452643871307373, 5.113150596618652,
  1.753124475479126, 2.1969480514526367, 2.578873872756958, 0.990979015827179, 0.1609385907649994, 0.23091404139995575,
  0.320975124835968, 0.1279768943786621, 0.11813713610172272, 0.016767939552664757, 17.050596237182617, 18.09605598449707,
  5.831860542297363, 5.082934379577637, 2.5815889835357666, 4.703032493591309, 2.696244239807129, 1.0099971294403076,
  1.6901977062225342, 1.759400725364685, 0.7901315093040466, 0.5735861659049988, 0.5544204115867615, 0.38350972533226013,
  0.35669323801994324, 0.3160327970981598,
];

// AmplitudeToDB(stype="magnitude") of that: multiplier 20, not 10.
const MEL_MAGNITUDE_HTK_DB = [
  24.56841468811035, 22.2569637298584, 13.926370620727539, 19.042102813720703, 19.620412826538086, 16.695261001586914,
  11.991653442382812, 12.224409103393555, 7.980354309082031, 0.15228523313999176, -13.919624328613281, -11.03087043762207,
  -4.4346160888671875, -9.219158172607422, -16.897796630859375, -24.119476318359375, 22.304086685180664, 22.910964965820312,
  17.78022575378418, 18.091304779052734, 7.3859333992004395, 15.650996208190918, 4.891705513000488, 6.040451526641846,
  -0.3270420432090759, -7.239322662353516, -16.61382484436035, -10.08210563659668, -8.097478866577148, -12.990142822265625,
  -17.854097366333008, -28.58217430114746, 22.799121856689453, 22.478153228759766, 18.244239807128906, 18.653141021728516,
  8.779973983764648, 17.14016342163086, 7.217329978942871, 10.35983657836914, 9.579968452453613, 1.10580575466156,
  -15.411080360412598, -7.2835283279418945, -3.5420541763305664, -12.941631317138672, -21.992130279541016, -34.135066986083984,
  22.90113067626953, 24.094356536865234, 19.375431060791016, 17.610149383544922, 8.449381828308105, 14.173770904541016,
  4.876255035400391, 6.836395740509033, 8.228601455688477, -0.07871083170175552, -15.866795539855957, -12.730993270874023,
  -9.870573043823242, -17.85736846923828, -18.552270889282227, -35.510406494140625, 24.634790420532227, 25.15167999267578,
  15.316143035888672, 14.12229061126709, 8.23774242401123, 13.44756031036377, 8.615184783935547, 0.0864027887582779,
  4.558750152587891, 4.907295227050781, -2.0460124015808105, -4.82802677154541, -5.123215675354004, -8.324472427368164,
  -8.954102516174316, -10.00535774230957,
];

// The librosa flavour of the same clip: MelSpectrogram(norm="slaney",
// mel_scale="slaney"), which agrees with librosa.feature.melspectrogram on its
// own defaults to 1.7e-5 relative. Two hundred times smaller than the htk/none
// numbers above, from the same audio.
const MEL_POWER_SLANEY = [
  1.3886805772781372, 0.3048657178878784, 0.05495874956250191, 0.18730923533439636, 0.11199554800987244, 0.05602080002427101,
  0.03354436531662941, 0.0075217969715595245, 0.004416218027472496, 0.001279371790587902, 9.969810344045982e-05, 4.317445473134285e-06,
  1.5348112356150523e-05, 1.1247876500419807e-05, 1.1946370932491845e-06, 1.0814893158794803e-07, 0.9782465696334839, 0.36909833550453186,
  0.2710469365119934, 0.09627458453178406, 0.00291745038703084, 0.030797002837061882, 0.018768593668937683, 0.0014227812644094229,
  0.001230187132023275, 0.0001767514186212793, 2.4277718694065697e-05, 4.414284831000259e-06, 1.2700646948360372e-05, 3.1666791073803324e-06,
  4.847684067499358e-07, 8.807425189161222e-08, 1.0272958278656006, 0.32706308364868164, 0.3003016710281372, 0.10978380590677261,
  0.004488005768507719, 0.04701842740178108, 0.022314846515655518, 0.0028176093474030495, 0.0031630471348762512, 0.001765446737408638,
  0.00010648631723597646, 6.102700808696682e-06, 2.4933564418461174e-05, 7.630549589521252e-06, 2.7864660978593747e-07, 2.5041488882493468e-08,
  1.1521477699279785, 0.5196648836135864, 0.2537013590335846, 0.08423930406570435, 0.0053486754186451435, 0.021429065614938736,
  0.013771097175776958, 0.0010108525166288018, 0.001716847880743444, 0.001149265212006867, 8.633190736873075e-05, 3.6681105939351255e-06,
  5.530727321456652e-06, 2.0729676180053502e-06, 3.7309163758436625e-07, 5.7582603574246605e-08, 1.639589786529541, 0.6435562372207642,
  0.06439276784658432, 0.07135635614395142, 0.008534683845937252, 0.012201812118291855, 0.01846783608198166, 0.0011166328331455588,
  0.0003983902861364186, 0.0011477514635771513, 0.0002692830457817763, 3.238057252019644e-05, 2.0484345441218466e-05, 9.685557415650692e-06,
  2.637994384713238e-06, 1.1668539627862629e-06,
];

/** The clip as torchaudio.load would hand it over: int16 divided by 32768. */
const WAVE = Float64Array.from(CLIP, (sample) => sample / 32768);
const SPECTROGRAM = stft({ input: WAVE, nFft: A_NFFT, hop: A_HOP, window: hannWindow(A_NFFT) });

function melOf(overrides: Partial<Parameters<typeof melSpectrogram>[0]> & {
  scale?: "htk" | "slaney";
  norm?: "none" | "slaney";
} = {}) {
  const { scale = "htk", norm = "none", ...rest } = overrides;
  const filterbank = melFilterbank({ nMels: A_MELS, nFft: A_NFFT, sampleRate: A_SR, scale, norm });
  return melSpectrogram({
    real: SPECTROGRAM.real,
    imag: SPECTROGRAM.imag,
    frames: A_FRAMES,
    bins: A_BINS,
    filterbank,
    nMels: A_MELS,
    ...rest,
  });
}

describe("mel / reference", () => {
  it("puts the STFT it is fed at the shape torchaudio does", () => {
    expect([SPECTROGRAM.frames, SPECTROGRAM.bins]).toEqual([A_FRAMES, A_BINS]);
  });

  it("converts Hz to mel on both scales, as torchaudio does", () => {
    expect(agree(PROBE_HZ.map((hz) => hzToMel(hz, "htk")), PROBE_MEL_HTK, SCALE)).toBeNull();
    expect(agree(PROBE_HZ.map((hz) => hzToMel(hz)), PROBE_MEL_HTK, SCALE)).toBeNull();
    expect(agree(PROBE_HZ.map((hz) => hzToMel(hz, "slaney")), PROBE_MEL_SLANEY, SCALE)).toBeNull();
    // Not two units for one quantity: at 4 kHz they are 2146 and 35.2.
    expect(hzToMel(4000, "htk") / hzToMel(4000, "slaney")).toBeGreaterThan(60);
  });

  it("inverts itself on both scales, including across Slaney's 1 kHz knee", () => {
    for (const scale of ["htk", "slaney"] as const) {
      expect(agree(PROBE_HZ.map((hz) => melToHz(hzToMel(hz, scale), scale)), PROBE_HZ, SCALE)).toBeNull();
    }
    // The knee itself: below it the map is linear at 200/3 Hz per mel, above it
    // exponential, and the two halves meet at 1000 Hz rather than merely nearly.
    expect(melToHz(15, "slaney")).toBeCloseTo(1000, 12);
    // One ulp below the knee is still on the linear side: 200/3 * 14.999999.
    expect(melToHz(14.999999, "slaney")).toBeCloseTo(999.9999333333333, 9);
  });

  it("spaces the band edges the way librosa.mel_frequencies does", () => {
    expect(agree(melFrequencies({ nMels: 8, fMax: 8000, scale: "htk" }), EDGES_HTK, SCALE)).toBeNull();
    expect(agree(melFrequencies({ nMels: 8, fMax: 8000, scale: "slaney" }), EDGES_SLANEY, SCALE)).toBeNull();
    // nMels + 2, because each band needs a left edge, a peak and a right edge.
    expect(melFrequencies({ nMels: 8, fMax: 8000 }).length).toBe(10);
  });

  it("builds torchaudio's default filterbank: htk scale, unnormalised", () => {
    const fb = melFilterbank({ nMels: FB_NMELS, nFft: FB_NFFT, sampleRate: FB_SR });
    expect(fb.length).toBe(FB_NMELS * FB_BINS);
    expect(agree(fb, FB_HTK_NONE, FILTERBANK)).toBeNull();
    // Unnormalised means the triangles reach 1 where a bin lands on a peak, and
    // just under it where none does.
    expect(Math.max(...fb)).toBeLessThanOrEqual(1);
    expect(Math.max(...fb)).toBeGreaterThan(0.98);
  });

  it("builds librosa's default filterbank: slaney scale, slaney normalisation", () => {
    const fb = melFilterbank({
      nMels: FB_NMELS, nFft: FB_NFFT, sampleRate: FB_SR, scale: "slaney", norm: "slaney",
    });
    expect(agree(fb, FB_SLANEY_SLANEY, FILTERBANK_NORMALISED)).toBeNull();
    // Area-normalised, so nothing is near 1 at all.
    expect(Math.max(...fb)).toBeLessThan(0.002);
  });

  it("treats the scale and the normalisation as separate choices", () => {
    const htkSlaney = melFilterbank({
      nMels: FB_NMELS, nFft: FB_NFFT, sampleRate: FB_SR, scale: "htk", norm: "slaney",
    });
    const slaneyNone = melFilterbank({
      nMels: FB_NMELS, nFft: FB_NFFT, sampleRate: FB_SR, scale: "slaney", norm: "none",
    });
    const row = (fb: Float32Array) => fb.subarray(2 * FB_BINS, 3 * FB_BINS);
    expect(agree(row(htkSlaney), FB_ROW2_HTK_SLANEY, FILTERBANK_NORMALISED)).toBeNull();
    expect(agree(row(slaneyNone), FB_ROW2_SLANEY_NONE, FILTERBANK)).toBeNull();
    // Neither is the other's default in disguise: the same band starts three
    // bins apart on the two scales.
    expect(agree(row(htkSlaney), row(slaneyNone), { rel: 1e-3, abs: 1e-3 })).not.toBeNull();
  });

  it("notices a band that caught no bin", () => {
    const dense = melFilterbank({ nMels: 40, nFft: FB_NFFT, sampleRate: FB_SR });
    expect(melFilterbankHasEmptyBand(dense, 40, FB_BINS)).toBe(true);
    const sparse = melFilterbank({ nMels: FB_NMELS, nFft: FB_NFFT, sampleRate: FB_SR });
    expect(melFilterbankHasEmptyBand(sparse, FB_NMELS, FB_BINS)).toBe(false);
  });

  it("matches torchaudio.transforms.MelSpectrogram on real audio", () => {
    expect(agree(melOf(), MEL_POWER_HTK, MEL)).toBeNull();
  });

  it("matches AmplitudeToDB(stype='power') on real audio", () => {
    expect(agree(melOf({ db: true }), MEL_POWER_HTK_DB, MEL_DB)).toBeNull();
  });

  it("matches MelSpectrogram(power=1) and AmplitudeToDB(stype='magnitude')", () => {
    expect(agree(melOf({ power: 1 }), MEL_MAGNITUDE_HTK, MEL)).toBeNull();
    expect(agree(melOf({ power: 1, db: true }), MEL_MAGNITUDE_HTK_DB, MEL_DB)).toBeNull();
    // The magnitude feature is not the square root of the power one — the
    // filterbank sums before the root in one and after it in the other.
    expect(MEL_MAGNITUDE_HTK[0]!).toBeLessThan(Math.sqrt(MEL_POWER_HTK[0]!) - 1);
  });

  it("matches the librosa flavour when asked for it", () => {
    expect(agree(melOf({ scale: "slaney", norm: "slaney" }), MEL_POWER_SLANEY, MEL)).toBeNull();
  });

  it("floors the log where torchaudio floors it, and not by shifting", () => {
    const silence = new Float32Array(A_FRAMES * A_BINS);
    const quiet = (power: 1 | 2) =>
      melSpectrogram({
        real: silence, imag: silence, frames: A_FRAMES, bins: A_BINS, nMels: A_MELS,
        filterbank: melFilterbank({ nMels: A_MELS, nFft: A_NFFT, sampleRate: A_SR }),
        power, db: true,
      });
    // amplitude_to_DB(0, multiplier, amin=1e-10, db_multiplier=0) in torchaudio.
    expect([...quiet(2)].every((v) => v === -100)).toBe(true);
    expect([...quiet(1)].every((v) => v === -200)).toBe(true);
    // Clamping the argument leaves everything above the floor untouched, where
    // log(x + amin) would move all of it. The loudest band here is 325.5, and
    // its dB is the dB of 325.5 exactly, not of 325.5 + 1e-10.
    expect(melOf({ db: true })[0]).toBeCloseTo(10 * Math.log10(MEL_POWER_HTK[0]!), 4);
  });

  it("refuses shapes that do not line up", () => {
    const filterbank = melFilterbank({ nMels: A_MELS, nFft: A_NFFT, sampleRate: A_SR });
    const args = {
      real: SPECTROGRAM.real, imag: SPECTROGRAM.imag, bins: A_BINS, filterbank, nMels: A_MELS,
    };
    expect(() => melSpectrogram({ ...args, frames: 4 })).toThrow(/real must be 4 x 129/);
    expect(() => melSpectrogram({ ...args, frames: A_FRAMES, nMels: 8 })).toThrow(/filterbank must be 8 x 129/);
    expect(() => melFrequencies({ nMels: 4, fMin: 8000, fMax: 100 })).toThrow(/fMax must exceed fMin/);
  });
});
