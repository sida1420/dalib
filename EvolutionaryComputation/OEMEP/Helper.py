

# --- FAST SCALAR HELPERS (No Object Creation) ---
def fast_lines_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """Returns True if line segment (p1,p2) intersects (p3,p4)"""
    # Orientation 1
    val1 = (p2y - p1y) * (p3x - p2x) - (p2x - p1x) * (p3y - p2y)
    val2 = (p2y - p1y) * (p4x - p2x) - (p2x - p1x) * (p4y - p2y)
    
    if (val1 > 0 and val2 > 0) or (val1 < 0 and val2 < 0):
        return False
        
    # Orientation 2
    val3 = (p4y - p3y) * (p1x - p4x) - (p4x - p3x) * (p1y - p4y)
    val4 = (p4y - p3y) * (p2x - p4x) - (p4x - p3x) * (p2y - p4y)
    
    if (val3 > 0 and val4 > 0) or (val3 < 0 and val4 < 0):
        return False
        
    return True


def fast_ray_segment_intersect(ray_ox, ray_oy, ray_dx, ray_dy, seg_p1x, seg_p1y, seg_p2x, seg_p2y):
    """Calculates T for ray intersection without creating objects"""
    rx = ray_dx
    ry = ray_dy
    sx = seg_p2x - seg_p1x
    sy = seg_p2y - seg_p1y

    down = rx * sy - ry * sx
    if abs(down) < 1e-9:
        return None

    dx = seg_p1x - ray_ox
    dy = seg_p1y - ray_oy

    t = (dx * sy - dy * sx) / down
    u = (dx * ry - dy * rx) / down

    if t <= 0 or t >= 1: # Ray bound (0 to 1)
        return None
    if u < 0 or u > 1:   # Segment bound
        return None
    return t
# ------------------------------------------------