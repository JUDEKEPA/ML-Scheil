%% Title: ML results validation class
% Author: Zhengdi Liu (lzdseu@gmail.com)
% Last update: 2024-05-19

classdef MLDataValidate < handle
    %MLDATAVALIDATE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        database;
        elemenets;
        allowed_phase;
        data_num;
        data_for_validation;
        all_validation_result = struct();
        valid_path = struct();
    end
    
    methods
        function obj = MLDataValidate(database, elements)
            %MLDATAVALIDATE Construct an instance of this class
            %   Detailed explanation goes here
            obj.database = database;
            obj.elemenets = elements;
        end
        
        function load_validation_data(obj, file_path)
            %LOAD_VALIDATION_DATA load format data
            
            file = fileread(file_path);
            obj.data_for_validation = jsondecode(file);
            
            obj.allowed_phase = obj.data_for_validation.allowed_phase;
            obj.data_num = obj.data_for_validation.data_num;
        end
        
        function validate_all(obj)
            % Integrated validation funciton. Validate all loaded data.
            for i = 0:obj.data_num - 1
                disp(i);
                single_path_data = getfield(obj.data_for_validation, ['path', num2str(i)]);
                [all_allowed, single_path_data_result] = obj.validate_single_path(single_path_data);
                obj.all_validation_result = setfield(obj.all_validation_result, ...
                                            ['path', num2str(i)], single_path_data_result);
                                        
                if all_allowed == true
                    obj.valid_path = setfield(obj.valid_path, ...
                                              ['path', num2str(i)], single_path_data_result);
                end
            end
        end
        
        function [all_allowed, single_path_data_result] = validate_single_path(obj, single_path_data)
            % Each designed path validation. Concentrate on if any not
            % allowed phase generated.
            single_path_data_result = struct();
            
            validate_points = single_path_data.validate_points;
            num_of_points = size(validate_points, 1);
            
            for i = 1:num_of_points
                each_point_data = struct();
                composition = validate_points(i, :);
                scheil_calc = ScheilCalculation(obj.database, obj.elemenets, composition);
                scheil_calc.calculate();
                scheil_calc.solid_overview();
                
                each_point_data.composition = composition;
                each_point_data.phases_in_solid = string(scheil_calc.phases_in_solid);
                each_point_data.phase_fracs_in_solid = cell2mat(scheil_calc.phase_fracs_in_solid);
                
                single_path_data_result = setfield(single_path_data_result, ...
                ['point', num2str(i)], each_point_data);
            end
            
            all_allowed = obj.if_all_allowed_phase(single_path_data_result, num_of_points);
            
        end
        
        function all_allowed = if_all_allowed_phase(obj, single_path_data_result, num_of_points)
            % Validate if only allowed phase generated.
            all_allowed = true;
            for i = 1:num_of_points
                point_data = getfield(single_path_data_result, ['point', num2str(i)]);
                phases_in_solid = point_data.phases_in_solid;
                
                for j = 1:length(phases_in_solid)
                    phase = phases_in_solid(j);
                    if ismember(phase, obj.allowed_phase)
                        continue
                    else
                        all_allowed = false;
                        break
                    end
                end
                
                if all_allowed == true
                    continue
                else
                    break
                end
            end
        end

    end
end

