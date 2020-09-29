/*
* 
* dependent-selects
* 
* Show filtered options on one select field depending on another
* See in action https://codepen.io/furalyon/pen/NzrXZL
*
* By Ramkishore Manorahan - @furalyon
* 
*
* To use:
* 1. Include this script
* 2. Use the markup format as shown in the example.html
*
* usage eg:

    Parent 1: <select name="parent" id="id_parent" data-child-id='id_child' class='dependent-selects__parent'>
        <option value="">--------</option>
        <option value="one" data-child-options="11|#twelve|#13">One</option>
        <option value="two" data-child-options="11|#14|#15">Two</option>
    </select>

    Child 1: <select name="child" id="id_child" class='dependent-selects__child'>
        <option value="">--------</option>
        <option value="11">Eleven</option>
        <option value="twelve">Twelve</option>
        <option value="13">Thirteen</option>
        <option value="14">fourteen</option>
        <option value="15">fifteen</option>
    </select>

*
* Note: A page can have multiple sets of this
*
*/


var handle_dependent_selects = function($parent) {
    var $child = document.getElementById($parent.getAttribute('data-child-id')),
        $selected = $parent.options[$parent.selectedIndex],
        parent_val = $selected.value;

    for (var i=0; i<$child.options.length; i++) {
        var $option = $child.options[i];
        if($option.value != '') {
            $option.setAttribute('hidden',true);
        }
    };

    if(parent_val) {
        var child_options = $selected.getAttribute('data-child-options'),
            child_options_array = child_options.split('|#');
        
        for (i=0; i<$child.options.length; i++) {
            var $option = $child.options[i];
            if ($option.value == "") {
                $option.innerText = "--------";
                continue;
            }
            if(child_options_array.indexOf($option.value) != -1) {
                $option.removeAttribute('hidden');
            }
        };

    } else {
        var show_text = $child.getAttribute('data-text-if-parent-empty');
        if(!show_text) {
            show_text = 'Select ' + $parent.name;
        }
        for (i=0; i<$child.options.length; i++) {
            var $option = $child.options[$child.selectedIndex];
            if ($option.value == "") {
                $option.innerText = '- ' + show_text + ' -';
                break;
            }
        };
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var $parents = document.getElementsByClassName('dependent-selects__parent');
    for (var i=0; i<$parents.length; i++) {
        handle_dependent_selects($parents[i]);
        $parents[i].addEventListener('change', function() {
            handle_dependent_selects(this)
        })
    }
}, false);