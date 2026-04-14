# 1. Create the Normalizer
    # This maps the input range [min_db, max_db] to [0.0, 1.0]
    min_db=-80.0
    max_db=0.0
    norm = mcolors.Normalize(vmin=min_db, vmax=max_db)
    
    # 2. Get the Colormap object from matplotlib
    cmap_name=cmap_name
    cmap = plt.get_cmap(cmap_name)
    
    # 3. Calculate the color
    # norm(db_value) converts dB to 0-1 scale
    # cmap(...) takes that 0-1 value and returns (R, G, B, A) in 0.0-1.0 floats
    rgba_color = cmap(norm(db_value))
    
    # 4. Return only RGB (first 3 values)
    # If you need 0-255 integers, multiply these by 255 and cast to int
    return rgba_color #[:3] # removing alpha remover