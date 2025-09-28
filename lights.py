from lifxlan import LifxLAN

# Number of lights you own (just 1 for now)
lifx = LifxLAN(1)

# Discover lights on LAN
lights = lifx.get_lights()
print("Found lights:", [l.get_label() for l in lights])

light = lights[0]  # grab the first

# Turn on
light.set_power("on")

# Set color: [Hue, Saturation, Brightness, Kelvin]
# Hue: 0–65535 (0 = red, 21845 = green, 43690 = blue)
# Saturation: 0–65535 (0 = white, 65535 = full color)
# Brightness: 0–65535
# Kelvin: 2500 (warm) – 9000 (cool), used for white tones
light.set_color([43690, 65535, 40000, 3500])  # bright blue

# Dim warm white
light.set_color([0, 0, 20000, 3500])