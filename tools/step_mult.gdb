# file; step_mult.gdb

define step_mult
	set $step_mult_max = 1000
	if $argc >= 1
		set @step_mult_max = $arg0
	endif

	set $step_mult_count = 0
	while ($step_mult_count < $step_mult_max)
		set $step_mult_count = $step_mult_count + 1
		printf "step #%d\n", $step_mult_count
		step
	end
end
