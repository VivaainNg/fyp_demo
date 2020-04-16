var selected_frame_num = document.getElementById("slider_id");
function show_video() {          
    document.getElementById(
        'vid_canvas').src = "/get_frame_num?frame_num=" 
        + selected_frame_num.value
        + "&random=" 
        + new Date().getTime();
}

//Display slider position
var show = document.getElementById("show_slider_val");
show.innerHTML = selected_frame_num.value;
selected_frame_num.oninput = function() {
    show.innerHTML = this.value;
}