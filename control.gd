extends Control



func _on_button_pressed():
	
	var img = Image.new()
	img = img.create(28, 28, false, Image.FORMAT_RGBA8)
	img.fill(Color.WHITE)
	
	for index in $GridContainer.get_child_count():
		var button = $GridContainer.get_child(index)
		if button.button_pressed:
			var row = (index / 28)
			var column = (index % 28)
			
			img.set_pixelv(Vector2i(column, row), Color.BLACK)
	
	img.save_png("C:/Users/Kuba/PycharmProjects/WandB/picture/image.png")
