extends Button


func _on_mouse_entered():
	if Input.is_action_pressed("LMB"):
		button_pressed = true
	elif Input.is_action_pressed("RMB"):
		button_pressed = false
