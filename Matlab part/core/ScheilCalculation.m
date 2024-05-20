%% Title: Scheil calculation class
% Author: Zhengdi Liu (lzdseu@gmail.com)
% Last update: 2024-05-19

classdef ScheilCalculation < handle
    %SCHEILCALCULATION is integrated Scheil-Gulliver model calculation
    %class
    %   Detailed explanation goes here
    
    properties
        TRIAL_START_TEMP = 1500;  % initial test temperature. 
        % Adjust according to estimated melting point.
        
        elements_cell;
        element_num;
        database;
        all_phase_name;
        liquid_index;
        origin_mole_fractions;
        
        temperature_ax;
        left_liquid = 1;
        solid_frac = 0; % sum of solid fraction
        solid_frac_ax; % solid fraction of each step
        composition_matrix; % all composition data
        phase_frac_matrix; % all phase fraction data
        
        % Post data process
        phases_in_solid;
        phase_fracs_in_solid;
        
    end
    
    methods
        function obj = ScheilCalculation(database, elements, mole_fracs)
            %SCHEILCALCULATION Construct an instance of this class
            %   elements is a char() 'fe co ni cr mn' 'fe c mn'
            %   mole_fracs is corresponding ele mole fraction
            %   [0.1, 0.2, 0.3, 0.1, 0.3]
        
            obj.database = database;
            elements = upper(elements);
            obj.elements_cell = regexp(elements, ' ', 'split');
            obj.element_num = size(obj.elements_cell, 2);
            obj.origin_mole_fractions = mole_fracs;
            
            tc_init_root; % Initiate the TC system
            tc_open_database(database); % Choose the database
            tc_check_error; % Check for error and reset them
            
            tc_element_select(elements);
            tc_get_data;
            
            [~, all_phase_name] = tc_list_phase();

            % Find indices of cells containing '#'
            contains_hash = contains(all_phase_name, '#');

            % Remove cells containing '#'
            obj.all_phase_name = transpose(all_phase_name(~contains_hash));
            obj.liquid_index = find(strcmp(obj.all_phase_name, 'LIQUID'));
            
            tc_set_condition('p', 101325);
            tc_set_condition('n', 1);
            
            for i = 1:size(mole_fracs, 2)-1
                tc_set_condition(['x(', cell2mat(obj.elements_cell(i)), ')'],...
                mole_fracs(i));
            end
        end
        
        function find_start_temp(obj)
            %FIND_START_TEMP use larger temperature step to fast find start
            %temperature
            %   The start temperature will be stored in
            %   obj.TRIAL_START_TEMP
            tc_set_condition('t', obj.TRIAL_START_TEMP);
            tc_compute_equilibrium;
            liquid_frac = tc_get_value('np(LIQUID)');
            
            % Initial temperature generate totally LIQUID
            if liquid_frac >= 0.999
                while 1
                    tc_set_condition('t', obj.TRIAL_START_TEMP - 20);
                    tc_compute_equilibrium;

                    liquid_frac = tc_get_value('np(LIQUID)');
                    if liquid_frac >= 0.999
                        obj.TRIAL_START_TEMP = obj.TRIAL_START_TEMP - 20;
                    else
                        break;
                    end
                end

            % Initial temperature generate contains solid phase
            else
                while 1
                    tc_set_condition('t', obj.TRIAL_START_TEMP + 20);
                    tc_compute_equilibrium;

                    liquid_frac = tc_get_value('np(LIQUID)');
                    if liquid_frac <= 0.999
                        obj.TRIAL_START_TEMP = obj.TRIAL_START_TEMP + 20;
                    else
                        obj.TRIAL_START_TEMP = obj.TRIAL_START_TEMP + 20;
                        break;
                    end
                end
            end
        end
        
        function calculate(obj)
            %CALCULATE is core function, executes whole scheil calculation
            %process.
            obj.find_start_temp();  % get start temperature 
            
            % Preallocate big enough array to store results.
            obj.composition_matrix = cell(obj.TRIAL_START_TEMP, obj.element_num*size(obj.all_phase_name, 2));
            obj.phase_frac_matrix = cell(obj.TRIAL_START_TEMP, size(obj.all_phase_name, 2));
            obj.solid_frac_ax = cell(obj.TRIAL_START_TEMP, 1);
            obj.temperature_ax = cell(obj.TRIAL_START_TEMP, 1);
            
            step = 1; % result row index
            
            temperature = obj.TRIAL_START_TEMP;
            all_liquid_comp = [];
            while 1
                tc_set_condition('t', temperature);
                tc_compute_equilibrium;
                obj.temperature_ax(step) = num2cell(temperature);
                
                [generated_phase_frac, comp_cell] = obj.gener_pha_frac();

                % Multiply left liquid to get real generatioin value
                generated_phase_frac = cellfun(@(x) x * obj.left_liquid,...
                generated_phase_frac, 'UniformOutput', false);

                % phase fraction and composition storage
                obj.phase_frac_matrix(step, :) = generated_phase_frac;
                obj.composition_matrix(step, :) = comp_cell;
                
                % Update solid and liquid fraction
                liquid_frac = tc_get_value('np(LIQUID)');
                obj.solid_frac = obj.solid_frac + obj.left_liquid * (1-liquid_frac);
                obj.left_liquid = obj.left_liquid * liquid_frac;
                obj.solid_frac_ax(step) = num2cell(obj.solid_frac);

                if obj.left_liquid <= 0.001
                    break;
                else
                    liquid_composition = obj.get_liquid_composition();
                    obj.reset_composition();  % next step calculation
                    all_liquid_comp = [all_liquid_comp;liquid_composition];
                    temperature = temperature - 1;  % default temperature step 1.
                end
                
                step = step + 1
            end
            
            % calculation finished, data storage, clip
            obj.composition_matrix = obj.composition_matrix(1:step, :);
            obj.phase_frac_matrix = obj.phase_frac_matrix(1:step, :);
            obj.solid_frac_ax = obj.solid_frac_ax(1:step, :);
            obj.temperature_ax = obj.temperature_ax(1:step, :);
            
            % clip zero columns
            [zeroColumns, obj.phase_frac_matrix] = obj.zero_delete(obj.phase_frac_matrix);
            obj.all_phase_name(zeroColumns) = [];
            [~, obj.composition_matrix] = obj.zero_delete(obj.composition_matrix);
            
            % get axis length
            for i = 1:size(obj.solid_frac_ax, 1)
                if cell2mat(obj.solid_frac_ax(i)) > 0.0005
                    break;
                end
            end
            
            obj.temperature_ax = obj.temperature_ax(i:end, :);
            obj.solid_frac_ax = obj.solid_frac_ax(i:end, :);
            obj.phase_frac_matrix = obj.phase_frac_matrix(i:end, :);
            obj.composition_matrix = obj.composition_matrix(i:end, :);
        end

        function reset_composition(obj)
            %RESET_COMPOSITION update composition condition setting with
            %the new liquid composition.
            liquid_composition = obj.get_liquid_composition();
            
            for i = 1:size(obj.elements_cell, 2)-1
                tc_set_condition(['x(', cell2mat(obj.elements_cell(i)), ')'],...
                liquid_composition(i));
            end
        end
        
        function liquid_composition = get_liquid_composition(obj)
            %GET_LIQUID_COMPOSITION
            liquid_composition = [];
            for i = 1:size(obj.elements_cell, 2)
                liquid_composition = [liquid_composition, ...
                    tc_get_value(['x(LIQUID,', cell2mat(obj.elements_cell(i)), ')'])];
            end
        end
        
        function [generated_phase_frac, comp_cell] = gener_pha_frac(obj)
            %GENER_PHA_FRAC get formatted phase fraction and corresponding
            %composition result.
            %   
            generated_phase_frac = cell(1, size(obj.all_phase_name, 2));
            generated_phase_frac(:) = {0};
            
            comp_cell = cell(1, size(obj.all_phase_name, 2)*obj.element_num);
            comp_cell(:) = {0};
            
            [phase_amount, all_phase_name] = tc_list_phase();
            phase_frac=zeros(phase_amount,1);
            i = 1;
            while(i<phase_amount+1)
                phase_frac(i) = obj.get_phase_frac(all_phase_name(i));
                i = i+1;
            end
            
            i=1;
            [m, ~] = size(all_phase_name);
            while(i<m+1)
                if phase_frac(i)>0
                    generated_phase = cell2mat(all_phase_name(i));
                    
                    if contains(generated_phase, '#')
                        generated_phase_nowell = generated_phase(1:strfind(temp, '#')-1);
                    else
                        generated_phase_nowell = generated_phase;
                    end
                    
                    phase_index = find(strcmp(obj.all_phase_name, generated_phase_nowell));
                    former_frac = cell2mat(generated_phase_frac(phase_index));
                    update_frac = former_frac + phase_frac(i);
                    generated_phase_frac(phase_index) = num2cell(update_frac);
                    
                    % Phase composition part
                    start_index = (phase_index-1) * obj.element_num + 1;
                    former_comp = cell2mat(comp_cell(start_index:start_index + obj.element_num - 1));
                    weighted_former = former_comp * former_frac;
                    
                    single_pha_comp = [];
                    for j = 1:obj.element_num
                        format_str = ['x(', generated_phase, ',', cell2mat(obj.elements_cell(j)), ')'];
                        comp = tc_get_value(format_str);
                        single_pha_comp = [single_pha_comp, comp];
                    end
                    weighted_temp = phase_frac(i)*single_pha_comp;
                    update_comp = (weighted_former + weighted_temp)/(former_frac + phase_frac(i));
                    comp_cell(start_index:start_index + obj.element_num - 1) = num2cell(update_comp);
                    
                end
                i=i+1;
            end
        end
        
        function solid_overview(obj)
            %SOLID_OVERVIEW get some information of solid.
            liquid_index = strcmp(obj.all_phase_name, 'LIQUID');
            obj.phases_in_solid = obj.all_phase_name;
            obj.phases_in_solid(liquid_index) = [];
            obj.phase_fracs_in_solid = cell(1, size(obj.phases_in_solid, 2));
            obj.phase_fracs_in_solid(:) = {0};
            
            for i = 1:size(obj.phases_in_solid, 2)
                phase = cell2mat(obj.phases_in_solid(i));
                phase_index = find(strcmp(obj.all_phase_name, phase));
                obj.phase_fracs_in_solid(i) = {sum(cell2mat(obj.phase_frac_matrix(:, phase_index)))};
            end
            
        end
        
        function [liquid_comp_mass, solid_comp_mass] = get_comp_mass(obj, liquid_composition, sys_liquid_frac)
            liquid_comp_mass = sys_liquid_frac * liquid_composition;
            solid_comp_mass = obj.origin_mole_fractions - liquid_comp_mass;
        end
        
        function draw_scheil_curve(obj)
            %DRAW_SCHEIL scheil solidification curve visualization
            label_cell = cell(size(obj.solid_frac_ax, 1), 1);
            for i = 1:size(obj.phase_frac_matrix, 1)
                path_temp = '';
                for j = 1:size(obj.all_phase_name, 2)
                    temp_phase = cell2mat(obj.all_phase_name(j));
                    if strcmp(temp_phase, 'LIQUID')
                    elseif cell2mat(obj.phase_frac_matrix(i, j)) > 1e-5
                        path_temp = [path_temp, cell2mat(obj.all_phase_name(j)), '+'];
                    end
                end
                path_temp = path_temp(1:end-1);
                label_cell(i) = cellstr(path_temp);
            end
            
            y_ax = cell2mat(obj.temperature_ax);
            x_ax = cell2mat(obj.solid_frac_ax);
            

            % Define a colormap or custom color mapping
            % Get unique labels
            labelList = unique(label_cell);
            colorMap = lines(length(labelList));

            hold on;

            % Plot points section by section based on labels
            for i = 1:length(labelList)
                currentLabel = labelList{i};
                mask = strcmp(label_cell, currentLabel);

                % Find segments of consecutive points with the current label
                startIndices = [];
                endIndices = [];
                
                % link all points
                flag = 0;
                for j = 1:length(mask)
                    if mask(j) == 1
                        if flag == 0
                            startIndices = [startIndices, j];
                            flag = 1;
                        elseif flag == 1
                            if j == length(mask)
                                endIndices = [endIndices, j];
                            else
                                continue
                            end
                        end
                    elseif mask(j) == 0
                        if flag == 1 
                            endIndices = [endIndices, j];
                            flag = 0;
                        else
                            continue
                        end
                    end
                end

                % Plot each contiguous segment
                for j = 1:length(startIndices)
                    idxRange = startIndices(j):endIndices(j);
                    plot(x_ax(idxRange), y_ax(idxRange), 'Color', ...
                    colorMap(i, :), 'DisplayName', ...
                    strrep(currentLabel, '_', '\_'), ...
                    'LineWidth', 2);
                end
            end

            hold off;
            
            legend;
            xlabel('Solid fraction');
            ylabel('Temperature');

        end
        
        function draw_comp_change_in_phase(obj, phase, elements_arr, x_axis, y_axis)
            % Visualization composition change in assigned phase. Show the
            % microsegregation.
            phase_index = find(strcmp(obj.all_phase_name, {phase}));
            start_index = obj.element_num * (phase_index - 1) + 1;
            end_index = start_index + obj.element_num - 1;
            phase_composition = obj.composition_matrix(:, ...
                                start_index:end_index);
            
            row_to_keep = true(size(phase_composition, 1), 1);
           
            for i = 1:size(phase_composition, 1)
                row = phase_composition{i};
                if all(row == 0)
                    row_to_keep(i) = false;
                end
            end
            
            t_ax = cell2mat(obj.temperature_ax(row_to_keep));
            s_frac_ax = cell2mat(obj.solid_frac_ax(row_to_keep));
            
            if strcmpi(x_axis, 'composition')
                comp_to_x = true;
                comp_to_y = false;
            elseif strcmpi(y_axis, 'composition')
                comp_to_y = true;
                comp_to_x = false;
            end
            
            if strcmpi(x_axis, 'temperature')
                x_ax = t_ax;
            elseif strcmpi(y_axis, 'temperature')
                y_ax = t_ax;
            end
            
            if strcmpi(x_axis, 'solid fraction')
                x_ax = s_frac_ax;
            elseif strcmpi(y_axis, 'solid fraction')
                y_ax = s_frac_ax;
            end
            
            
            hold on;
            
            for i = 1:size(elements_arr, 2)
                comp_index = find(strcmpi(obj.elements_cell, ...
                                  elements_arr(i)));
                comp_ax = cell2mat(phase_composition(:, comp_index));
                
                if comp_to_x
                    x_ax = comp_ax;
                elseif comp_to_y
                    y_ax = comp_ax;
                end
                
                plot(x_ax, y_ax, 'DisplayName', obj.elements_cell{comp_index}, ...
                    'LineWidth', 2);
            end
            
            hold off;
            
            legend;
            xlabel(x_axis);
            ylabel(y_axis);
        end
        
        function integrate_storage(obj, path)
        	% This function is used for storage of core data, easier for
        	% interaction for different programming language.
            path = string(path);
            data_name = "";
            for i = 1:size(obj.elements_cell, 2)
                data_name = data_name + string(obj.elements_cell{i}) + string(obj.origin_mole_fractions(i));
            end
            status = mkdir(path + "\" + data_name);
            if status
                data.elements = string(obj.elements_cell);
                data.all_phase_name = string(obj.all_phase_name);
                data.origin_mole_fraction = obj.origin_mole_fractions;
                data.temperature_ax = cell2mat(obj.temperature_ax);
                data.solid_frac_ax = cell2mat(obj.solid_frac_ax);
                data.composition_matrix = cell2mat(obj.composition_matrix);
                data.phase_frac_matrix = cell2mat(obj.phase_frac_matrix);
            end
            json_data = jsonencode(data);
            file = fopen(path + "\" + data_name + "\" + "data.json", 'w+');
            fprintf(file, '%s', json_data);
            fclose(file);
        end
        
    end
    
    
    methods(Static)
        function phase_frac = get_phase_frac(phase_name)
            name_modified = ['np(', char(phase_name), ')'];
            phase_frac = tc_get_value(name_modified);
        end
        
        function [zeroColumns, zero_col_cell] = zero_delete(zero_col_cell)
            % Logical array where true indicates a zero
            isZero = cellfun(@(x) isnumeric(x) && x == 0, zero_col_cell);

            % Find columns where all entries are zero
            zeroColumns = all(isZero, 1);
            
            % Remove columns that are entirely zero
            zero_col_cell(:, zeroColumns) = [];
       end
    end
end

