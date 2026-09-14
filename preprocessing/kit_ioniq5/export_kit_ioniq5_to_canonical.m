function export_kit_ioniq5_to_canonical(rawDir, outDir)
%EXPORT_KIT_IONIQ5_TO_CANONICAL Convert KIT IONIQ 5 MAT runs to canonical HDF5.
%
% Raw dataset:
%   Schulz, Tobias; Snobar, Fadi (2026)
%   DOI 10.35097/44a91t97pmnha1k9
%
% Usage:
%   export_kit_ioniq5_to_canonical( ...
%       'C:\path\to\data\dataset', ...
%       'C:\path\to\project\data\processed\kit_ioniq5');
%
% Why MATLAB here?
% The source MAT files store signals as MATLAB `timeseries` objects. scipy.io
% intentionally exposes those as opaque MCOS objects. The source-specific
% conversion is therefore done once in MATLAB. All estimation afterwards is
% source-independent Python.

arguments
    rawDir (1,1) string
    outDir (1,1) string
end

if ~isfolder(rawDir)
    error('Raw dataset folder not found: %s', rawDir);
end
if ~isfolder(outDir)
    mkdir(outDir);
end

% Load the supplied static vehicle/tire parameters.
run(fullfile(rawDir, 'parameter.m'));

files = dir(fullfile(rawDir, '*.mat'));
for k = 1:numel(files)
    inFile = fullfile(files(k).folder, files(k).name);
    S = load(inFile, 'data');
    if ~isfield(S, 'data')
        continue;
    end
    d = S.data;

    % The current 2-DoF estimator needs these signals. Other MAT runs are still
    % useful later (standstill/noise/steering studies), but are skipped here.
    required = {'delta_stm_rad','a_x_ra_mps2','a_y_ra_mps2','w_z_cor_radps', ...
                'v_x_cor_mps','v_y_cor_mps'};
    if ~all(isfield(d, required))
        fprintf('skip %-45s (not a dynamic 2-DoF run)\n', files(k).name);
        continue;
    end

    [~, stem, ~] = fileparts(files(k).name);
    outFile = fullfile(outDir, stem + ".h5");
    if isfile(outFile)
        delete(outFile);
    end

    % Canonical time origin: steering signal start. Original signals are
    % synchronous at 1000 Hz, but we still store one time vector per signal.
    t0 = firstTime(d.delta_stm_rad);

    % First write creates the HDF5 file; root attributes can follow.
    writeTs(outFile, 'input.steering_angle', d.delta_stm_rad, t0, ...
        'input', 'rad', 'vehicle', 'delta_stm_rad');

    h5writeatt(outFile, '/', 'schema_version', '1.0');
    h5writeatt(outFile, '/', 'source_dataset', ...
        'KIT Multi-Surface Driving Maneuvers DOI 10.35097/44a91t97pmnha1k9');
    h5writeatt(outFile, '/', 'source_file', files(k).name);
    h5writeatt(outFile, '/', 'coordinate_convention', 'ISO8855 x-forward y-left z-up');
    h5writeatt(outFile, '/', 'nominal_sample_rate_hz', 1000.0);

    % Longitudinal acceleration is kept at its actual sensor location.
    % The 3-DoF model transforms rear-axle acceleration to the CoG internally.
    writeTs(outFile, 'input.longitudinal_acceleration_rear_axle', d.a_x_ra_mps2, t0, ...
        'input', 'm/s^2', 'rear_axle', 'a_x_ra_mps2');

    % Yaw rate is a rigid-body angular velocity and therefore independent of
    % the translational sensor location.
    writeTs(outFile, 'sensor.yaw_rate', d.w_z_cor_radps, t0, ...
        'measurement', 'rad/s', 'vehicle', 'w_z_cor_radps');

    % Keep the lateral acceleration in its true physical frame. The Python
    % estimator uses a rear-axle measurement equation rather than pretending
    % this is CoG acceleration.
    writeTs(outFile, 'sensor.lateral_acceleration_rear_axle', d.a_y_ra_mps2, t0, ...
        'measurement', 'm/s^2', 'rear_axle', 'a_y_ra_mps2');

    % Transform Correvit velocity to the CoG using rigid-body kinematics.
    % parameter.m defines translation vectors base_frame -> target_frame.
    r_cor_to_cog = params.tf.trvec_cor_ra + params.tf.trvec_ra_cog;
    r_cog_to_cor = -r_cor_to_cog;

    [tVx, vxCor] = unpackTs(d.v_x_cor_mps, t0);
    [tVy, vyCor] = unpackTs(d.v_y_cor_mps, t0);
    [tR,  yawR]  = unpackTs(d.w_z_cor_radps, t0);

    % Use the Correvit time grid for the derived reference signals.
    vyCorI = interp1(tVy, vyCor, tVx, 'linear', 'extrap');
    yawRI  = interp1(tR,  yawR,  tVx, 'linear', 'extrap');

    xS = r_cog_to_cor(1);
    yS = r_cog_to_cor(2);
    % v_sensor = v_CoG + omega x r_CoG->sensor
    vxCog = vxCor + yawRI .* yS;
    vyCog = vyCorI - yawRI .* xS;
    betaCog = atan2(vyCog, vxCog);

    % Keep vx as an input for the existing 2-DoF KF and also expose the
    % same physical signal as a measurement for the 3-DoF EKF.
    writeArray(outFile, 'input.longitudinal_velocity', tVx, vxCog, ...
        'input', 'm/s', 'cog', 'derived from v_x_cor_mps + w_z_cor_radps');
    writeArray(outFile, 'sensor.longitudinal_velocity', tVx, vxCog, ...
        'measurement', 'm/s', 'cog', 'derived from v_x_cor_mps + w_z_cor_radps');

    writeTruth(outFile, 'beta_rad', tVx, betaCog, ...
        'rad', 'cog', 'derived from Correvit planar velocity and yaw rate');
    writeTruth(outFile, 'longitudinal_velocity_mps', tVx, vxCog, ...
        'm/s', 'cog', 'derived from Correvit planar velocity and yaw rate');
    writeTruth(outFile, 'lateral_velocity_mps', tVx, vyCog, ...
        'm/s', 'cog', 'derived from Correvit planar velocity and yaw rate');

    % Preserve useful source channels for later work without feeding them to
    % the current KF. `kind=aux` means the generic adapter ignores them.
    writeTs(outFile, 'aux.correvit_vx', d.v_x_cor_mps, t0, ...
        'aux', 'm/s', 'correvit', 'v_x_cor_mps');
    writeTs(outFile, 'aux.correvit_vy', d.v_y_cor_mps, t0, ...
        'aux', 'm/s', 'correvit', 'v_y_cor_mps');
    if isfield(d, 'F_trl_N')
        writeTs(outFile, 'aux.tie_rod_force_left', d.F_trl_N, t0, ...
            'aux', 'N', 'tie_rod_left', 'F_trl_N');
    end
    if isfield(d, 'F_trr_N')
        writeTs(outFile, 'aux.tie_rod_force_right', d.F_trr_N, t0, ...
            'aux', 'N', 'tie_rod_right', 'F_trr_N');
    end

    % Static model parameters are duplicated as metadata for provenance.
    h5writeatt(outFile, '/', 'vehicle_mass_kg', params.vehicle.m);
    h5writeatt(outFile, '/', 'yaw_inertia_kgm2', params.vehicle.I);
    h5writeatt(outFile, '/', 'lf_m', params.vehicle.l_f);
    h5writeatt(outFile, '/', 'lr_m', params.vehicle.l_r);
    h5writeatt(outFile, '/', 'Cf_N_per_rad', params.tire.C_f);
    h5writeatt(outFile, '/', 'Cr_N_per_rad', params.tire.C_r);

    fprintf('wrote %-44s -> %s\n', files(k).name, outFile);
end
end

function t0 = firstTime(ts)
    t = normalizeTime(ts.Time);
    t0 = t(1);
end

function [t, y] = unpackTs(ts, t0)
    t = normalizeTime(ts.Time) - t0;
    y = squeeze(double(ts.Data));
    y = y(:);
    t = t(:);
    if numel(t) ~= numel(y)
        error('Timeseries length mismatch.');
    end
end

function t = normalizeTime(rawTime)
    if isduration(rawTime)
        t = seconds(rawTime);
    elseif isdatetime(rawTime)
        t = seconds(rawTime - rawTime(1));
    else
        t = double(rawTime);
    end
    t = t(:);
end

function writeTs(file, source, ts, t0, kind, unit, frame, rawSource)
    [t, y] = unpackTs(ts, t0);
    writeArray(file, source, t, y, kind, unit, frame, rawSource);
end

function writeArray(file, source, t, y, kind, unit, frame, rawSource)
    base = "/signals/" + source;
    createAndWrite(file, base + "/time_s", t(:));
    createAndWrite(file, base + "/value", y(:));
    h5writeatt(file, base, 'kind', kind);
    h5writeatt(file, base, 'unit', unit);
    h5writeatt(file, base, 'frame', frame);
    h5writeatt(file, base, 'raw_source', rawSource);
end

function writeTruth(file, stateName, t, y, unit, frame, sourceDescription)
    base = "/truth/" + stateName;
    createAndWrite(file, base + "/time_s", t(:));
    createAndWrite(file, base + "/value", y(:));
    h5writeatt(file, base, 'unit', unit);
    h5writeatt(file, base, 'frame', frame);
    h5writeatt(file, base, 'source', sourceDescription);
end

function createAndWrite(file, path, value)
    h5create(file, path, size(value), 'Datatype', 'double');
    h5write(file, path, double(value));
end
